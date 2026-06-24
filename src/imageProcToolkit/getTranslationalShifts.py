import numpy as np

# The phase-correlation primitives (normalized cross-power surface, 3x3 quad-fit
# sub-pixel peak, wraparound pick, and the cupy GPU gate) are shared with
# getSimilarityTransform.py's Fourier-Mellin path. They live in _phaseCorrelationCore
# so the sign-sensitive logic is not duplicated; re-bind them here so this module's
# public surface (including the private names callers may have imported) is unchanged.
from ._phaseCorrelationCore import (
    _phaseCorrelationMap, _quadFit3x3, _peakAndSubpixel, _wraparoundPick)

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


# --------------------------------------------------------------------------- #
# pairwise phase correlation (primitives live in _phaseCorrelationCore)
# --------------------------------------------------------------------------- #
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