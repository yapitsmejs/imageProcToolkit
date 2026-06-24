import numpy as np
from .fftUpsample import fourierUpsample

'''
    Estimate the relative translational shift between every image pair via phase
    correlation, then solve a global least-squares for a per-image shift vector
    (dy_i, dx_i) with **no image as ground truth** (zero-mean gauge). This runs on the
    uint8 amplitude images from step 3 (estimation only -- writes nothing to disk).

    --------------------------------------------------------------------------- #
    Shift convention

    pairwiseTranslationalShift(A, B) returns s_AB = the translation to apply to **B** to align it to
    **A** (A = master, B = slave: F(A) * conj(F(B)), peak -> shift applied to B).

    Residual after applying per-image shifts t_i:  res_ij = s_ij - (t_j - t_i).
    Minimizing sum res_ij^2 gives the normal equations  **L t = -d**  with the unweighted
    graph Laplacian (L_ii = deg_i, L_ij = -1) and  d_k = sum_{k<j} s_kj - sum_{i<k} s_ik.
    NOTE the minus sign: setting the gradient d/dt_k (sum res^2) = 0 yields (L t)_k = -d_k,
    not +d_k (verified on a 2-node example: one edge s_01=5, zero-mean gauge gives
    t=(-2.5, 2.5), and L t = -d). Driving the residual to zero => t_j - t_i = s_ij.

    The gauge (global translation is unobservable, L is singular) is fixed by sum t = 0
    via the augmented system [[L, 1],[1.T, 0]] @ [t, lam] = [-d, 0], solved with lstsq.
    This distributes the correction across all images -- the "no ground-truth image"
    requirement.

    The synthetic self-check (known ground-truth shifts) is the arbiter.

'''

# cupy is an optional GPU fast-path for the pairwise phase-correlation FFTs, mirroring
# geometricCorrectionPFA.py's guarded switch. A usable GPU is not required: detect at
# import whether a CUDA device is actually present and fall back to the NumPy FFT path
# everywhere if not. cupy imports fine without a GPU/driver; the failure surfaces only
# when querying devices, so we guard getDeviceCount and treat 0/exception as "no GPU".
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

    Inputs are the uint8 images from step 3 (cast to float32). NaN / valid handling
    replaces the reference's np.nan_to_num(..., nan=nanmin(...)) (temp/_registration.py:
    106-107) which manufactures a fake dark border that itself correlates: the valid
    region is the *intersection* of the two masks; each image is mean-removed over its
    valid pixels and zero-filled outside before the FFT.

    Epsilon guard fixes the reference's unguarded /= np.abs(crossPower)
    (temp/_registration.py:114) which is 0/0 -> NaN on zero bins: the cross-power is
    normalized by (|crossPower| + eps) with eps scaled to its own max.

    Uses rfft2 / irfft2 (real input -> half-spectrum, ~2x faster than the reference's
    fft2). The cupy path mirrors geometricCorrectionPFA's "move once, compute on GPU,
    bring back once" pattern.

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
    rows, matching temp/_registration.py:140). Peak: solve the 2x2 first-order system;
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
    around the integer peak (temp/_registration.py:124-149). If upsampleFactor > 1 the
    3x3 patch is first upsampled (fourierUpsample per axis) and the quadratic fit run on
    the upsampled 3x3 neighborhood around the upsampled peak; the offset is mapped back
    through the upsample factor. Default 1 (off) -- the quadratic fit alone matches the
    reference."""
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
    to zero element-wise (temp/_registration.py:151-152). Returns (dy, dx) (rows/y,
    columns/x)."""
    truePeak = np.asarray(truePeak, dtype=np.float64)
    wrappedPeak = truePeak - np.asarray(shape, dtype=np.float64)
    shift = np.where(np.abs(truePeak) < np.abs(wrappedPeak), truePeak, wrappedPeak)
    return float(shift[0]), float(shift[1])


def pairwiseTranslationalShift(imgA, imgB, maskA=None, maskB=None, subpixel=True, upsampleFactor=1):
    """Translation to apply to imgB to align it to imgA (s_AB), via phase correlation.

    Args:
        imgA, imgB: real images (uint8 from step 3; cast to float32 internally).
        maskA, maskB: optional boolean valid-pixel masks (intersection is correlated).
        subpixel: if True, refine the integer peak with the 2-D quadratic fit.
        upsampleFactor: if > 1, refine via patch upsampling before the fit (default 1).

    Returns (dy, dx, peakHeight, confidence): (dy, dx) is s_AB (rows/y, columns/x);
    peakHeight ~= 1 for a clean peak; confidence is peakHeight (diagnostic only -- not
    used as a weight, per the unweighted-solve decision)."""
    corrMap, _ = _phaseCorrelationMap(imgA, imgB, maskA, maskB)
    if subpixel:
        truePeak, peakHeight = _peakAndSubpixel(corrMap, upsampleFactor)
    else:
        peakIdx = np.unravel_index(int(np.nanargmax(corrMap)), corrMap.shape)
        truePeak = np.array(peakIdx, dtype=np.float64)
        peakHeight = float(corrMap[peakIdx])
    dy, dx = _wraparoundPick(truePeak, corrMap.shape)
    return dy, dx, peakHeight, float(peakHeight)


def allPairwiseTranslationalShifts(images, masks=None, subpixel=True, upsampleFactor=1):
    """Phase-correlation shift for every i < j pair (A=i, B=j).

    Returns dict keyed by (i, j) with i < j -> (dy, dx, peakHeight, confidence). Only the
    i < j half is stored (s_ji = -s_ij is derivable). `masks` is an optional parallel
    list of boolean arrays (one per image)."""
    n = len(images)
    pw = {}
    for i in range(n):
        mA = masks[i] if masks is not None else None
        for j in range(i + 1, n):
            mB = masks[j] if masks is not None else None
            dy, dx, ph, conf = pairwiseTranslationalShift(images[i], images[j], mA, mB,
                                              subpixel, upsampleFactor)
            pw[(i, j)] = (dy, dx, ph, conf)
    return pw


# --------------------------------------------------------------------------- #
# global least-squares solve (zero-mean gauge)
# --------------------------------------------------------------------------- #
def solveGlobalTranslationalShifts(pairwise, n):
    """Per-image shifts t_i = (dy_i, dx_i) from the pairwise shifts, zero-mean gauge.

    Unweighted least-squares: minimize sum_ij (s_ij - (t_j - t_i))^2 per coordinate. The
    normal equations are L t = -d (see the module docstring for the sign), where L is
    the unweighted graph Laplacian (L_ii = deg_i, L_ij = -1 per observed edge i<j) and
    d_k = sum_{k<j} s_kj - sum_{i<k} s_ik. The global translation is unobservable (L is
    singular), so the zero-mean gauge sum t = 0 is fixed via the augmented system
    [[L, 1],[1.T, 0]] @ [t, lam] = [-d, 0], solved with lstsq. This distributes the
    correction across all images -- no image is ground truth.

    Args:
        pairwise: dict (i, j) -> (dy, dx, peakHeight, confidence), i < j.
        n:        number of images.

    Returns t of shape (n, 2), row order = node index, columns (dy, dx)."""
    L = np.zeros((n, n), dtype=np.float64)
    d = np.zeros((n, 2), dtype=np.float64)
    for (i, j), (dy, dx, _ph, _conf) in pairwise.items():
        s = np.array([dy, dx], dtype=np.float64)        # s_ij (apply to j to align to i)
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0
        # d_k = sum_{k<j} s_kj - sum_{i<k} s_ik: edge (i<j) contributes +s_ij to d_i,
        # -s_ij to d_j.
        d[i] += s
        d[j] -= s

    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[:n, :n] = L
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.zeros((n + 1, 2), dtype=np.float64)
    rhs[:n, :] = -d                                  # L t = -d
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return sol[:n, :]


def checkTranslationalShiftResiduals(pairwise, shifts):
    """Per-pair residual res_ij = s_ij - (t_j - t_i); report max / mean over both coords.

    Args:
        pairwise: dict (i, j) -> (dy, dx, peakHeight, confidence), i < j.
        shifts:   (n, 2) array of per-image shifts t_i.

    Returns dict with residualMax_px, residualMean_px, nPairs."""
    res = []
    for (i, j), (dy, dx, _ph, _conf) in pairwise.items():
        s = np.array([dy, dx], dtype=np.float64)
        pred = np.asarray(shifts[j], dtype=np.float64) - np.asarray(shifts[i], dtype=np.float64)
        res.append(s - pred)
    if res:
        res = np.array(res, dtype=np.float64)
        return {'residualMax_px': float(np.max(np.abs(res))),
                'residualMean_px': float(np.mean(np.abs(res))),
                'nPairs': len(pairwise)}
    return {'residualMax_px': 0.0, 'residualMean_px': 0.0, 'nPairs': len(pairwise)}


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def getTranslationalShifts(images, masks=None, subpixel=True, upsampleFactor=1):
    """Estimate per-image shifts (N, 2) for N images via all-pairwise phase correlation
    + a zero-mean-gauge global least-squares.

    Args:
        images: list of N real images (uint8 from step 3).
        masks:  optional list of N boolean valid-pixel masks.
        subpixel, upsampleFactor: forwarded to pairwiseTranslationalShift.

    Returns t of shape (N, 2), row order = input order, columns (dy, dx). Also prints
    the checkTranslationalShiftResiduals diagnostics to stdout."""
    pw = allPairwiseTranslationalShifts(images, masks, subpixel, upsampleFactor)
    t = solveGlobalTranslationalShifts(pw, len(images))
    diag = checkTranslationalShiftResiduals(pw, t)
    print(f"[getTranslationalShifts] {diag['nPairs']} pairs | "
          f"residualMax={diag['residualMax_px']:.4f}px  "
          f"residualMean={diag['residualMean_px']:.4f}px")
    return t
