"""Shared phase-correlation primitives used by both the translation estimator
(``getTranslationalShifts.py``) and the similarity estimator
(``getSimilarityTransform.py`` -- the log-polar Fourier-Mellin path phase-correlates
two spectra with the exact same normalized cross-power + sub-pixel peak machinery).

Extracted verbatim from ``getTranslationalShifts.py`` so the sign-sensitive quad-fit
and wraparound logic is not duplicated. The public surface of
``getTranslationalShifts.py`` is unchanged: it re-binds these four names at import
time (``from ._phaseCorrelationCore import ...``), so callers that imported the
private names from ``getTranslationalShifts`` keep working.

The cupy GPU gate lives here because ``_phaseCorrelationMap`` is the only consumer;
the modules that import these helpers do not reference ``_HAVE_CUPY_GPU`` / ``cp``
themselves (the dispatch is internal to ``_phaseCorrelationMap``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fftUpsample import fourierUpsample

__all__ = [
    "_phaseCorrelation_core",
    "_phaseCorrelationMap",
    "_quadFit3x3",
    "_peakAndSubpixel",
    "_wraparoundPick",
]

# cupy is an optional GPU fast-path for the pairwise phase-correlation FFTs. A
# usable GPU is not required: detect at import whether a CUDA device is actually
# present and fall back to the NumPy FFT path everywhere if not. cupy imports fine
# without a GPU/driver; the failure surfaces only when querying devices, so we
# guard getDeviceCount and treat 0/exception as "no GPU". `cp` is typed ``Any``
# because cupy mirrors numpy's API but is not stub-typed; the array-module
# dispatch (``xp`` param) is likewise ``Any``.
cp: Any = None
_HAVE_CUPY_GPU: bool = False
try:  # pragma: no cover  -- cupy not installed on CPU CI; try body is un-runnable there
    import cupy as _cp

    cp = _cp
    _HAVE_CUPY_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAVE_CUPY_GPU = False


# --------------------------------------------------------------------------- #
# pairwise phase correlation
# --------------------------------------------------------------------------- #
def _phaseCorrelation_core(
    A: np.ndarray, B: np.ndarray, shape: tuple[int, int], xp: Any
) -> np.ndarray:
    """``xp``-parameterized normalized cross-power phase-correlation FFT step.

    Args:
        A: first image (uint8 from step 3, cast to float32 by the caller).
        B: second image (uint8 from step 3, cast to float32 by the caller).
        shape: output shape ``(H, W)`` for ``irfft2``.
        xp: array module (``np`` or ``cp``).

    Returns:
        The non-negative correlation surface ``|irfft2(R)|`` on ``xp``'s backend.

    Notes:
        rfft2 cross-power of A,B, normalize by (|crossPower| + eps) with eps
        scaled to the cross-power max, irfft2 back to ``shape``, return ``|.``.
        The mean-removal, mask intersection, and valid-count are host-side NumPy
        (run by the caller in ``_phaseCorrelationMap``); this core is the only
        array-module step that differs between backends. ``xp.asarray`` moves
        A,B to the device on the cupy path (no-op on NumPy). The eps-max
        readback (``float(xp.abs(crossPower).max())``) syncs the GPU to the host
        for one scalar -- the same host round-trip the old mirrored cupy branch
        did.
    """
    a = xp.asarray(A)
    b = xp.asarray(B)
    crossPower = xp.fft.rfft2(a) * xp.conj(xp.fft.rfft2(b))
    eps = 1e-12 * (float(xp.abs(crossPower).max()) or 1.0)
    R = crossPower / (xp.abs(crossPower) + eps)
    result: np.ndarray = xp.abs(xp.fft.irfft2(R, s=shape))
    return result


def _phaseCorrelationMap(
    imgA: np.ndarray,
    imgB: np.ndarray,
    maskA: np.ndarray | None = None,
    maskB: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Normalized cross-power phase-correlation surface of two real images.

    Args:
        imgA: first real image (uint8 from step 3; cast to float32 internally).
        imgB: second real image (uint8 from step 3; cast to float32 internally).
        maskA: optional boolean valid-pixel mask for ``imgA`` (defaults to
            all-True).
        maskB: optional boolean valid-pixel mask for ``imgB`` (defaults to
            all-True).

    Returns:
        ``(corrMap, nValid)``: ``corrMap`` = ``|irfft2(R)|`` (non-negative, peak
        ~1 for a clean peak); ``nValid`` is the intersected valid-pixel count.

    Notes:
        NaN / valid handling: the valid region is the *intersection* of the two
        masks; each image is mean-removed over its valid pixels and zero-filled
        outside before the FFT (filling invalid pixels with the nanmin would
        manufacture a fake dark border that itself correlates). Epsilon guard:
        the cross-power is normalized by (|crossPower| + eps) with eps scaled to
        its own max, so a zero bin does not produce 0/0 -> NaN. Uses rfft2 /
        irfft2 (real input -> half-spectrum, ~2x faster than fft2). The cupy path
        keeps data on the GPU across the FFT pair ("move once, compute on GPU,
        bring back once").
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

    # Auto-dispatch the FFT step on `_HAVE_CUPY_GPU` (numpy in -> numpy out: the
    # mean-removal / masking preprocessing is host-side NumPy either way, so the
    # GPU only accelerates the rfft2/irfft2 pair via the xp-parameterized core
    # and the result is brought back to NumPy).
    if _HAVE_CUPY_GPU:  # pragma: no cover  -- GPU path, excluded from CPU CI coverage
        print("Using cupy for phase correlation")
        result: np.ndarray = cp.asnumpy(_phaseCorrelation_core(A, B, A.shape, cp))
        cp.get_default_memory_pool().free_all_blocks()
        corrMap = result
    else:
        corrMap = _phaseCorrelation_core(A, B, A.shape, np)
    return corrMap, nValid


def _quadFit3x3(patch: np.ndarray) -> tuple[float, float]:
    """2-D quadratic (6-param paraboloid) fit on a 3x3 patch.

    Args:
        patch: ``(3, 3)`` array of correlation-surface values around the peak.

    Returns:
        ``(dyOff, dxOff)``: the sub-pixel row/column offset of the fitted peak.
        ``(0, 0)`` if the fit is degenerate (``|denom| < 1e-12``).

    Notes:
        ``z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f``, with x the column offset and
        y the row offset (``meshgrid(arange(-1,2), arange(-1,2))`` -> X varies
        along columns, Y along rows). Peak: solve the 2x2 first-order system;
        ``denom = c^2 - 4*a*b``, ``dx = (2*b*d - c*e)/denom``,
        ``dy = (2*a*e - c*d)/denom``.
    """
    px, py = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    M = np.stack((px**2, py**2, px * py, px, py, np.ones_like(px)), axis=-1).reshape(-1, 6)
    coeffs, *_ = np.linalg.lstsq(M, patch.reshape(-1), rcond=None)
    a, b, c, d, e, _f = coeffs
    denom = c * c - 4.0 * a * b
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    dxOff = (2.0 * b * d - c * e) / denom
    dyOff = (2.0 * a * e - c * d) / denom
    return float(dyOff), float(dxOff)


def _peakAndSubpixel(
    corrMap: np.ndarray, upsampleFactor: int | float = 1
) -> tuple[np.ndarray, float]:
    """Locate the integer peak of ``corrMap`` and refine it to sub-pixel.

    Args:
        corrMap: 2-D correlation surface.
        upsampleFactor: if > 1, refine via patch upsampling before the fit
            (default 1 -- the quadratic fit alone is used).

    Returns:
        ``(truePeak, peakHeight)``: ``truePeak`` = ``[row, col]`` float peak
        location (rows/y, columns/x); ``peakHeight`` = ``corrMap`` at the
        integer peak (~1 for a clean peak -- the diagnostic "confidence").
    """
    peakIdx = np.unravel_index(int(np.nanargmax(corrMap)), corrMap.shape)
    peakHeight = float(corrMap[peakIdx])

    if upsampleFactor and upsampleFactor > 1:
        u = int(upsampleFactor)
        subR = np.arange(peakIdx[0] - 1, peakIdx[0] + 2)
        subC = np.arange(peakIdx[1] - 1, peakIdx[1] + 2)
        patch = np.take(
            np.take(corrMap, subR, axis=0, mode="wrap"), subC, axis=1, mode="wrap"
        ).astype(np.float32)
        patchU = fourierUpsample(fourierUpsample(patch, u, axis=1), u, axis=0)
        pu = np.unravel_index(int(np.argmax(patchU)), patchU.shape)
        ssR = np.arange(pu[0] - 1, pu[0] + 2)
        ssC = np.arange(pu[1] - 1, pu[1] + 2)
        patch3 = np.take(np.take(patchU, ssR, axis=0, mode="wrap"), ssC, axis=1, mode="wrap")
        dyU, dxU = _quadFit3x3(patch3)
        # the 3-pixel span (peakIdx-1 .. peakIdx+1) maps to upsampled index 0..3u;
        # an upsampled-patch position p corresponds to corrMap coord
        # peakIdx - 1 + p/u.
        truePeak = np.array(
            [peakIdx[0] - 1 + (pu[0] + dyU) / u, peakIdx[1] - 1 + (pu[1] + dxU) / u],
            dtype=np.float64,
        )
    else:
        subR = np.arange(peakIdx[0] - 1, peakIdx[0] + 2)
        subC = np.arange(peakIdx[1] - 1, peakIdx[1] + 2)
        patch = np.take(np.take(corrMap, subR, axis=0, mode="wrap"), subC, axis=1, mode="wrap")
        dyOff, dxOff = _quadFit3x3(patch)
        truePeak = np.array([peakIdx[0] + dyOff, peakIdx[1] + dxOff], dtype=np.float64)
    return truePeak, peakHeight


def _wraparoundPick(truePeak: np.ndarray, shape: tuple[int, ...]) -> tuple[float, float]:
    """Resolve the periodic phase-correlation peak to a signed shift.

    Args:
        truePeak: ``[row, col]`` float peak location (may be in ``[0, N)``).
        shape: ``(H, W)`` shape of the correlation surface.

    Returns:
        ``(dy, dx)`` (rows/y, columns/x) in ``[-N/2, N/2)``.

    Notes:
        ``wrappedPeak = truePeak - shape``; pick whichever of ``truePeak`` /
        ``wrappedPeak`` is closer to zero element-wise.
    """
    truePeak = np.asarray(truePeak, dtype=np.float64)
    wrappedPeak = truePeak - np.asarray(shape, dtype=np.float64)
    shift = np.where(np.abs(truePeak) < np.abs(wrappedPeak), truePeak, wrappedPeak)
    return float(shift[0]), float(shift[1])


def _phaseCorrelationMap_cupy_selfcheck() -> bool:
    """Equivalence check: cupy phase-correlation vs the NumPy reference.

    The NumPy reference is produced by calling the ``xp``-parameterized
    ``_phaseCorrelation_core`` with ``np`` directly (bypassing
    ``_phaseCorrelationMap``'s auto-dispatch, which uses the GPU whenever one is
    present). The GPU result comes from the public ``_phaseCorrelationMap``
    (numpy in -> numpy out, auto round-tripped through the GPU). cuFFT and NumPy's
    FFT use different backends, so the float32 correlation surfaces agree to
    ~1e-5 relative but not bit-for-bit; we check allclose, not array_equal, and
    also that the integer peak lands at the same pixel. Skipped (and reported as
    such) when no GPU.
    """
    if _HAVE_CUPY_GPU:  # pragma: no cover  -- GPU path, excluded from CPU CI coverage
        rng = np.random.default_rng(0)
        H, W = 64, 48
        base = (rng.standard_normal((H, W))).astype(np.float32)
        # circularly shift base by (dy, dx) to build imgB; the phase-correlation peak
        # should land at that shift (mod shape) -- exercises the FFT pair end to end.
        dy, dx = 5, -3
        imgB = np.roll(base, (dy, dx), axis=(0, 1))
        imgA = base

        # NumPy reference: reproduce _phaseCorrelationMap's host-side preprocessing,
        # then force the NumPy core (bypasses auto-dispatch, which would otherwise
        # compute on the GPU when one is present).
        A = np.where(np.isfinite(imgA), imgA - float(imgA.mean()), 0.0).astype(np.float32)
        B = np.where(np.isfinite(imgB), imgB - float(imgB.mean()), 0.0).astype(np.float32)
        ref = np.abs(_phaseCorrelation_core(A, B, A.shape, np))
        # GPU via the public function: numpy in -> numpy out.
        got, nValid = _phaseCorrelationMap(imgA, imgB)
        nio_ok = isinstance(got, np.ndarray) and isinstance(ref, np.ndarray)

        shape_ok = ref.shape == got.shape == (H, W)
        dtype_ok = got.dtype == ref.dtype == np.float32
        vals_close = np.allclose(got, ref, rtol=1e-3, atol=1e-4)
        # integer peak must agree (the sub-pixel estimator starts from this pixel).
        peak_ref = np.unravel_index(int(np.argmax(ref)), ref.shape)
        peak_got = np.unravel_index(int(np.argmax(got)), got.shape)
        peak_ok = peak_ref == peak_got
        max_diff = float(np.nanmax(np.abs(got - ref)))

        ok = bool(shape_ok and dtype_ok and vals_close and peak_ok and nio_ok)
        print(
            f"phaseCorrelationMap self-check: {'PASS' if ok else 'FAIL'} "
            f"(shape={shape_ok}, dtype={dtype_ok}, values_close={vals_close}, "
            f"peak_match={peak_ok}, nio={nio_ok}, max_abs_diff={max_diff:.3e})"
        )
        return ok

    print("phaseCorrelationMap self-check: skipped (no GPU available — NumPy path only).")
    return True


if __name__ == "__main__":  # pragma: no cover
    _phaseCorrelationMap_cupy_selfcheck()
