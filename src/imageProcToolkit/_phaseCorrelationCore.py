import numpy as np
from .fftUpsample import fourierUpsample

'''
    Shared phase-correlation primitives used by both the translation estimator
    (getTranslationalShifts.py) and the similarity estimator
    (getSimilarityTransform.py -- the log-polar Fourier-Mellin path phase-correlates
    two spectra with the exact same normalized cross-power + sub-pixel peak machinery).

    Extracted verbatim from getTranslationalShifts.py so the sign-sensitive quad-fit
    and wraparound logic is not duplicated. The public surface of
    getTranslationalShifts.py is unchanged: it re-binds these four names at import
    time (from ._phaseCorrelationCore import ...), so callers that imported the
    private names from getTranslationalShifts keep working.

    The cupy GPU gate lives here because _phaseCorrelationMap is the only consumer;
    the modules that import these helpers do not reference _HAVE_CUPY_GPU / cp
    themselves (the dispatch is internal to _phaseCorrelationMap).
'''

# cupy is an optional GPU fast-path for the pairwise phase-correlation FFTs. A usable
# GPU is not required: detect at import whether a CUDA device is actually present and
# fall back to the NumPy FFT path everywhere if not. cupy imports fine without a
# GPU/driver; the failure surfaces only when querying devices, so we guard
# getDeviceCount and treat 0/exception as "no GPU".
try:
    import cupy as cp
    try:
        _HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAVE_CUPY_GPU = False
except ImportError:
    cp = None
    _HAVE_CUPY_GPU = False


# --------------------------------------------------------------------------- #
# pairwise phase correlation
# --------------------------------------------------------------------------- #
def _phaseCorrelationMap(imgA, imgB, maskA=None, maskB=None):
    """Normalized cross-power phase-correlation surface of two real images.

    Inputs are the uint8 images from step 3 (cast to float32). NaN / valid handling:
    the valid region is the *intersection* of the two masks; each image is mean-removed
    over its valid pixels and zero-filled outside before the FFT. (Filling invalid
    pixels with the nanmin would manufacture a fake dark border that itself correlates.)

    Epsilon guard: the cross-power is normalized by (|crossPower| + eps) with eps
    scaled to its own max, so a zero bin does not produce 0/0 -> NaN.

    Uses rfft2 / irfft2 (real input -> half-spectrum, ~2x faster than fft2). The cupy
    path keeps data on the GPU across the FFT pair ("move once, compute on GPU, bring
    back once").

    Returns (corrMap, nValid): corrMap = |irfft2(R)| (non-negative, peak ~1 for a clean
    peak); nValid is the intersected valid-pixel count (coverage metric).
    """
    A = np.asarray(imgA, dtype=np.float32)
    B = np.asarray(imgB, dtype=np.float32)
    if maskA is None:
        maskA = np.ones(A.shape, dtype=bool)
    if maskB is None:
        maskB = np.ones(B.shape, dtype=bool)
    valid = np.asarray(maskA, dtype=bool) & np.asarray(maskB, dtype=bool)
    nValid = int(valid.sum())

    # mean-remove over valid pixels, zero-fill invalid pixels
    if nValid > 0:
        mA = float(A[valid].mean(dtype=np.float64))
        mB = float(B[valid].mean(dtype=np.float64))
    else:
        mA = mB = 0.0
    A = np.where(valid, (A - mA), 0.0).astype(np.float32)
    B = np.where(valid, (B - mB), 0.0).astype(np.float32)

    if _HAVE_CUPY_GPU:
        Ag = cp.asarray(A)
        Bg = cp.asarray(B)
        crossPower = cp.fft.rfft2(Ag) * cp.conj(cp.fft.rfft2(Bg))
        eps = 1e-12 * (float(cp.abs(crossPower).max()) or 1.0)
        R = crossPower / (cp.abs(crossPower) + eps)
        corrMap = np.abs(cp.asnumpy(cp.fft.irfft2(R, s=A.shape)))
        del Ag, Bg, crossPower, R
        cp.get_default_memory_pool().free_all_blocks()
    else:
        crossPower = np.fft.rfft2(A) * np.conj(np.fft.rfft2(B))
        eps = 1e-12 * (float(np.abs(crossPower).max()) or 1.0)
        R = crossPower / (np.abs(crossPower) + eps)
        corrMap = np.abs(np.fft.irfft2(R, s=A.shape))
    return corrMap, nValid


def _quadFit3x3(patch):
    """2-D quadratic (6-param paraboloid) fit on a 3x3 patch, returns (dyOff, dxOff).

    z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f, with x the column offset and y the row
    offset (meshgrid(arange(-1,2), arange(-1,2)) -> X varies along columns, Y along
    rows). Peak: solve the 2x2 first-order system;
    denom = c^2 - 4*a*b, dx = (2*b*d - c*e)/denom, dy = (2*a*e - c*d)/denom. Returns
    (0, 0) if |denom| < 1e-12 (degenerate / flat peak)."""
    px, py = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    M = np.stack((px ** 2, py ** 2, px * py, px, py, np.ones_like(px)),
                 axis=-1).reshape(-1, 6)
    coeffs, *_ = np.linalg.lstsq(M, patch.reshape(-1), rcond=None)
    a, b, c, d, e, f = coeffs
    denom = c * c - 4.0 * a * b
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    dxOff = (2.0 * b * d - c * e) / denom
    dyOff = (2.0 * a * e - c * d) / denom
    return float(dyOff), float(dxOff)


def _peakAndSubpixel(corrMap, upsampleFactor=1):
    """Locate the integer peak of corrMap and refine it to sub-pixel.

    Returns (truePeak, peakHeight): truePeak = [row, col] float peak location (rows/y,
    columns/x); peakHeight = corrMap at the integer peak (~1 for a clean peak -- the
    diagnostic "confidence").

    The primary sub-pixel method is the 2-D quadratic fit on the 3x3 mode='wrap' patch
    around the integer peak. If upsampleFactor > 1 the 3x3 patch is first upsampled
    (fourierUpsample per axis) and the quadratic fit run on the upsampled 3x3
    neighborhood around the upsampled peak; the offset is mapped back through the
    upsample factor. Default 1 (off) -- the quadratic fit alone is used."""
    peakIdx = np.unravel_index(int(np.nanargmax(corrMap)), corrMap.shape)
    peakHeight = float(corrMap[peakIdx])

    if upsampleFactor and upsampleFactor > 1:
        u = int(upsampleFactor)
        subR = np.arange(peakIdx[0] - 1, peakIdx[0] + 2)
        subC = np.arange(peakIdx[1] - 1, peakIdx[1] + 2)
        patch = np.take(np.take(corrMap, subR, axis=0, mode='wrap'),
                        subC, axis=1, mode='wrap').astype(np.float32)
        patchU = fourierUpsample(fourierUpsample(patch, u, axis=1), u, axis=0)
        pu = np.unravel_index(int(np.argmax(patchU)), patchU.shape)
        ssR = np.arange(pu[0] - 1, pu[0] + 2)
        ssC = np.arange(pu[1] - 1, pu[1] + 2)
        patch3 = np.take(np.take(patchU, ssR, axis=0, mode='wrap'),
                        ssC, axis=1, mode='wrap')
        dyU, dxU = _quadFit3x3(patch3)
        # the 3-pixel span (peakIdx-1 .. peakIdx+1) maps to upsampled index 0..3u; an
        # upsampled-patch position p corresponds to corrMap coord peakIdx - 1 + p/u.
        truePeak = np.array([peakIdx[0] - 1 + (pu[0] + dyU) / u,
                             peakIdx[1] - 1 + (pu[1] + dxU) / u], dtype=np.float64)
    else:
        subR = np.arange(peakIdx[0] - 1, peakIdx[0] + 2)
        subC = np.arange(peakIdx[1] - 1, peakIdx[1] + 2)
        patch = np.take(np.take(corrMap, subR, axis=0, mode='wrap'),
                        subC, axis=1, mode='wrap')
        dyOff, dxOff = _quadFit3x3(patch)
        truePeak = np.array([peakIdx[0] + dyOff, peakIdx[1] + dxOff], dtype=np.float64)
    return truePeak, peakHeight


def _wraparoundPick(truePeak, shape):
    """Resolve the periodic phase-correlation peak to a signed shift in [-N/2, N/2).

    wrappedPeak = truePeak - shape; pick whichever of truePeak / wrappedPeak is closer
    to zero element-wise. Returns (dy, dx) (rows/y, columns/x)."""
    truePeak = np.asarray(truePeak, dtype=np.float64)
    wrappedPeak = truePeak - np.asarray(shape, dtype=np.float64)
    shift = np.where(np.abs(truePeak) < np.abs(wrappedPeak), truePeak, wrappedPeak)
    return float(shift[0]), float(shift[1])