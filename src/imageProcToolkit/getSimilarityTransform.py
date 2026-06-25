import numpy as np
from ._phaseCorrelationCore import (
    _phaseCorrelationMap, _quadFit3x3, _peakAndSubpixel, _wraparoundPick)
from .interp2 import interp2linear
from . import getTranslationalShifts
from .clampImageAmplitude import clampImageAmplitude
from .normalizeImageAmplitude import normalizeImagesAmplitude
from .similarityTransformImage import similarityRotScaleImage

'''
    Estimate the relative similarity transform (rotation theta + uniform scale s, then
    translation dy/dx) between every image pair, then solve a global least-squares for
    a per-image similarity G_i = (theta_i, log s_i, dy_i, dx_i) with **no image as
    ground truth** (zero-mean gauge). This generalizes getTranslationalShifts from the
    2-DOF translation group to the 4-DOF similarity group SIM(2), and runs on the uint8
    amplitude images from step 3 (estimation only -- writes nothing to disk).

    --------------------------------------------------------------------------- #
    Why two stages (decoupled rot/scale, then translation)

    The translation global solve in getTranslationalShifts is linear because translation
    is an abelian additive group: the pairwise residual s_ij - (t_j - t_i) is linear in
    t, giving the Laplacian normal equations L t = -d. The similarity group SIM(2) is
    non-abelian. With G_i = the similarity to apply to image i to align it to the
    reference and S_ij = G_i^{-1} . G_j (apply to j to align to i, matching the existing
    A=master / B=slave convention), the pairwise components are

        theta_ij    = theta_j - theta_i            # additive -- clean
        log s_ij    = log s_j - log s_i            # additive -- clean
        t_ij        = (1/s_i) R(-theta_i)(t_j - t_i)   # bilinear in (t, s, theta)

    Rotation and scale are exactly additive in Lie-algebra coords, so stage 1 solves them
    with the SAME linear Laplacian as getTranslationalShifts (2 columns, zero-mean gauge)
    -- no linearization error. Translation is the coupled part, so stage 2 first
    de-rotates + de-scales each estimation image by (theta_i, s_i) (physically removing
    the rot/scale) and then REUSES getTranslationalShifts verbatim on the de-warped uint8
    for (dy_i, dx_i), where the residual is again exactly additive. The final per-image
    similarity is G_i = (theta_i, s_i, t'_i) with t'_i from stage 2; applying rot/scale
    then translation (see similarityTransformImage) is the correct composition.

    Gauge: stage-1 zero-mean fixes the 2 rot/scale gauge DOF; stage-2 zero-mean fixes the
    2 translation gauge DOF. All 4 gauge DOF fixed, no image is ground truth. Dominant
    residual error is second-order: estimation error in (theta_i, s_i) leaves a small
    residual rot/scale in the de-warped images that biases the translation sub-solve.

    --------------------------------------------------------------------------- #
    Stage 1: Fourier-Mellin (log-polar phase correlation of magnitude spectra)

    pairwiseSimilarityTransform(A, B) returns (theta, log s, peakHeight, confidence) =
    the rotation + uniform scale to apply to B to align it to A. A translation becomes a
    linear phase in fft2, so |fft2| is translation-invariant; a rotation rotates the
    magnitude spectrum and a uniform scale radially scales it. Resampling |fft2| (fftshifted, high-passed) onto a log-polar grid (rho = log radius, phi = angle) turns
    rotation into a shift along phi and scale into a shift along rho, so phase-correlating
    the two log-polar spectra recovers (theta, log s) with the same normalized cross-power
    + sub-pixel peak machinery as translation (_phaseCorrelationCore).

    Sign convention (pinned by the 2-node self-check, the arbiter):

        theta   =  d_phi * (2*pi / nPhi)        # rotation to apply to B
        log s   = -d_rho * drho_per_px          # scale to apply to B (natural log)

    where (d_phi, d_rho) is the signed phase-correlation peak in log-polar pixels and
    drho_per_px = (log rmax - log rmin) / (nRho - 1).

    --------------------------------------------------------------------------- #
    Masks / NaN handling (same rule as the rest of the pipeline)

    The valid-pixel mask MUST come from np.isfinite of the ORIGINAL input, not from the
    uint8: normalizeToUint8 maps NaN -> 0 sentinel, so a NaN border is undetectable in
    the uint8. For the de-warped stage-2 sub-solve the mask is re-derived from
    np.isfinite of the de-warped float (the warp marks its out-of-source border NaN).

    The synthetic self-check (known ground-truth similarity) is the arbiter.
'''


# --------------------------------------------------------------------------- #
# log-polar Fourier-Mellin pairwise (rotation + scale only)
# --------------------------------------------------------------------------- #
def _spectrumPrep(img, mask):
    """Mean-remove a real image over its valid pixels and zero-fill outside (the same
    pre-FFT prep as _phaseCorrelationMap), returning a float32 array ready for |fft2|.

    A None mask defaults to np.isfinite(img) -- NOT all-True -- so a NaN border (e.g. a
    de-warped image's out-of-source edge) is excluded from the mean and zero-filled
    rather than poisoning it. The pipeline passes explicit masks from np.isfinite of the
    original input; this default just makes the pairwise robust when called directly on
    NaN-bearing floats.
    """
    A = np.asarray(img, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(A)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        mA = float(A[mask].mean(dtype=np.float64))
    else:
        mA = 0.0
    return np.where(mask, A - mA, 0.0).astype(np.float32)


def _logPolarGrid(H, W, nRho, nPhi, rmin):
    """Build the log-polar coordinate grids for an (H, W) fftshifted spectrum.

    Returns (gx, gy, drho_per_px) where gx/gy are (nPhi, nRho) float32 arrays of
    column/row source coordinates (phi = axis 0 / rows, rho = axis 1 / cols), centered
    at the image centre, with r in [rmin, rmax] (rmax = inscribed-circle radius, to
    avoid corner aliasing) and phi in [0, 2*pi). drho_per_px is the log-radius step.
    """
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    rmax = min(cy, cx, H - 1 - cy, W - 1 - cx)
    if rmax <= rmin:
        rmax = rmin + 1.0
    if nRho is None:
        nRho = max(H, W)
    if nPhi is None:
        nPhi = 360
    phi = np.linspace(0.0, 2.0 * np.pi, nPhi, endpoint=False)        # (nPhi,)
    rho = np.linspace(np.log(rmin), np.log(rmax), nRho)              # (nRho,)
    PH, RH = np.meshgrid(phi, rho, indexing='ij')                    # (nPhi, nRho)
    r = np.exp(RH)
    gx = (cx + r * np.cos(PH)).astype(np.float32)                    # col / x
    gy = (cy + r * np.sin(PH)).astype(np.float32)                    # row / y
    drho_per_px = (float(np.log(rmax)) - float(np.log(rmin))) / (nRho - 1)
    return gx, gy, drho_per_px


def _highPassSpectrum(M):
    """High-pass a fftshifted magnitude spectrum: log1p compresses dynamic range, a
    Gaussian high-pass suppresses the DC + low-frequency lobe (which otherwise pins the
    phase-correlation peak at (0,0) regardless of the true misalignment), and the DC
    pixel is zeroed. The sigma is ~8% of the shorter dimension -- suppress only the very
    low frequencies, keep the mid/high band that carries the rot/scale structure."""
    H, W = M.shape
    M = np.log1p(M)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.ogrid[:H, :W]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = 0.08 * min(H, W)
    hp = 1.0 - np.exp(-r2 / (2.0 * sigma * sigma))
    M = M * hp
    M[H // 2, W // 2] = 0.0          # zero the DC pixel
    return M


def pairwiseSimilarityTransform(imgA, imgB, maskA=None, maskB=None,
                                subpixel=True, upsampleFactor=1,
                                highPass=True, nRho=None, nPhi=None, rmin=1.0):
    """Rotation + uniform scale to apply to imgB to align it to imgA (S_ij^{rs}), via
    Fourier-Mellin (log-polar phase correlation of the magnitude spectra).

    Args:
        imgA, imgB: real images (uint8 from step 3; cast to float32 internally).
        maskA, maskB: optional boolean valid-pixel masks (intersection is prepped).
        subpixel: if True, refine the integer peak with the 2-D quadratic fit.
        upsampleFactor: if > 1, refine via patch upsampling before the fit (default 1).
        highPass: if True (default), high-pass the spectra so the low-freq lobe does not
            pin the peak at (0, 0). Strongly recommended.
        nRho, nPhi: log-polar grid sizes (default nRho = max(H, W), nPhi = 360).
        rmin: inner log-polar radius (default 1.0; smaller extends scale range but
            aliases near DC).

    Returns (theta, log s, peakHeight, confidence): (theta, log s) is the rotation (rad)
    and log-scale (natural log) to apply to B to align to A; peakHeight ~= 1 for a clean
    peak; confidence is peakHeight (diagnostic only -- not used as a weight)."""
    A = _spectrumPrep(imgA, maskA)
    B = _spectrumPrep(imgB, maskB)
    H, W = A.shape

    # translation-invariant magnitude spectra, DC centred
    MA = np.fft.fftshift(np.abs(np.fft.fft2(A)))
    MB = np.fft.fftshift(np.abs(np.fft.fft2(B)))
    if highPass:
        MA = _highPassSpectrum(MA)
        MB = _highPassSpectrum(MB)
    else:
        MA = np.log1p(MA)
        MB = np.log1p(MB)

    gx, gy, drho_per_px = _logPolarGrid(H, W, nRho, nPhi, rmin)
    logPolarA = interp2linear(MA, gx, gy, extrapval=np.nan)
    logPolarB = interp2linear(MB, gx, gy, extrapval=np.nan)

    # The log-polar resample leaves an out-of-source NaN border (interp2linear's
    # extrapval). _phaseCorrelationMap defaults its masks to all-True, so a single
    # NaN pixel in the border would enter the mean-removal and poison the entire
    # cross-power -> an all-NaN corrMap -> nanargmax raises "All-NaN slice". Pass
    # the finite-region masks so it mean-removes over the valid intersection and
    # zero-fills the border instead. (If the whole map is NaN the intersection is
    # empty -> crossPower 0 -> corrMap all-zero finite -> degenerate (0,0) peak,
    # not a crash.)
    corrMap, _ = _phaseCorrelationMap(logPolarA, logPolarB,
                                      maskA=np.isfinite(logPolarA),
                                      maskB=np.isfinite(logPolarB))
    if subpixel:
        truePeak, peakHeight = _peakAndSubpixel(corrMap, upsampleFactor)
    else:
        peakIdx = np.unravel_index(int(np.nanargmax(corrMap)), corrMap.shape)
        truePeak = np.array(peakIdx, dtype=np.float64)
        peakHeight = float(corrMap[peakIdx])
    d_phi, d_rho = _wraparoundPick(truePeak, corrMap.shape)

    # verified on a 2-node example: theta = d_phi * (2*pi/nPhi), log s = -d_rho * drho_per_px
    dphi_per_px = 2.0 * np.pi / corrMap.shape[0]
    theta = d_phi * dphi_per_px
    log_s = -d_rho * drho_per_px
    return theta, log_s, peakHeight, float(peakHeight)


def allPairwiseSimilarityTransforms(images, masks=None, subpixel=True, upsampleFactor=1,
                                    highPass=True, nRho=None, nPhi=None, rmin=1.0):
    """Fourier-Mellin rotation+scale for every i < j pair (A=i, B=j).

    Returns dict keyed by (i, j) with i < j -> (theta, log s, peakHeight, confidence).
    Only the i < j half is stored (the inverse is derivable). `masks` is an optional
    parallel list of boolean arrays (one per image)."""
    n = len(images)
    pw = {}
    for i in range(n):
        mA = masks[i] if masks is not None else None
        for j in range(i + 1, n):
            mB = masks[j] if masks is not None else None
            theta, log_s, ph, conf = pairwiseSimilarityTransform(
                images[i], images[j], mA, mB, subpixel, upsampleFactor,
                highPass, nRho, nPhi, rmin)
            pw[(i, j)] = (theta, log_s, ph, conf)
    return pw


# --------------------------------------------------------------------------- #
# global least-squares solve for rotation + scale (zero-mean gauge)
# --------------------------------------------------------------------------- #
def solveGlobalSimilarityRotScale(pairwise, n):
    """Per-image (theta_i, log s_i) from the pairwise rot/scale, zero-mean gauge.

    Unweighted least-squares: minimize sum_ij ((theta_ij - (theta_j - theta_i))^2 +
    (log s_ij - (log s_j - log s_i))^2). Rotation and scale are exactly additive in
    these Lie-algebra coordinates, so the normal equations are L p = -d with the same
    unweighted graph Laplacian as solveGlobalTranslationalShifts (L_ii = deg_i,
    L_ij = -1) and d_k = sum_{k<j} s_kj - sum_{i<k} s_ik. The global rotation+scale is
    unobservable (L is singular), so the zero-mean gauge sum p = 0 is fixed via the
    augmented system [[L, 1],[1.T, 0]] @ [p, lam] = [-d, 0], solved with lstsq.

    Args:
        pairwise: dict (i, j) -> (theta, log s, peakHeight, confidence), i < j.
        n:        number of images.

    Returns p of shape (n, 2), row order = node index, columns (theta, log s)."""
    L = np.zeros((n, n), dtype=np.float64)
    d = np.zeros((n, 2), dtype=np.float64)
    for (i, j), (theta_ij, logs_ij, _ph, _conf) in pairwise.items():
        s = np.array([theta_ij, logs_ij], dtype=np.float64)   # s_ij (apply to j to align to i)
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0
        d[i] += s
        d[j] -= s

    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[:n, :n] = L
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.zeros((n + 1, 2), dtype=np.float64)
    rhs[:n, :] = -d                                  # L p = -d
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return sol[:n, :]


def checkSimilarityRotScaleResiduals(pairwise, params):
    """Per-pair residual res_ij = (theta_ij - (theta_j - theta_i),
    log s_ij - (log s_j - log s_i)); report max / mean per component.

    Args:
        pairwise: dict (i, j) -> (theta, log s, peakHeight, confidence), i < j.
        params:   (n, 2) array of per-image (theta_i, log s_i).

    Returns dict with residualMax_rot_rad, residualMean_rot_rad,
    residualMax_scale_log, residualMean_scale_log, nPairs."""
    resRot, resScale = [], []
    for (i, j), (theta_ij, logs_ij, _ph, _conf) in pairwise.items():
        s = np.array([theta_ij, logs_ij], dtype=np.float64)
        pred = np.asarray(params[j], dtype=np.float64) - np.asarray(params[i], dtype=np.float64)
        diff = s - pred
        resRot.append(diff[0])
        resScale.append(diff[1])
    if resRot:
        resRot = np.array(resRot, dtype=np.float64)
        resScale = np.array(resScale, dtype=np.float64)
        return {'residualMax_rot_rad': float(np.max(np.abs(resRot))),
                'residualMean_rot_rad': float(np.mean(np.abs(resRot))),
                'residualMax_scale_log': float(np.max(np.abs(resScale))),
                'residualMean_scale_log': float(np.mean(np.abs(resScale))),
                'nPairs': len(pairwise)}
    return {'residualMax_rot_rad': 0.0, 'residualMean_rot_rad': 0.0,
            'residualMax_scale_log': 0.0, 'residualMean_scale_log': 0.0,
            'nPairs': len(pairwise)}


# --------------------------------------------------------------------------- #
# orchestrator: 4-DOF similarity per image (rot/scale, then translation)
# --------------------------------------------------------------------------- #
def getSimilarityTransform(images, masks=None, subpixel=True, upsampleFactor=1,
                           highPass=True, nRho=None, nPhi=None, rmin=1.0):
    """Estimate per-image similarity (N, 4) = (theta, log s, dy, dx), row order = input
    order, via two-stage decoupled estimation.

    Stage 1: all-pairwise Fourier-Mellin rotation+scale on the inputs -> a zero-mean-gauge
    global least-squares for (theta_i, log s_i). Stage 2: de-rotate + de-scale each input
    by (theta_i, s_i), then REUSE getTranslationalShifts on the de-warped uint8 for
    (dy_i, dx_i) (the de-warp physically removes the rot/scale, leaving a pure-translation
    residual that the existing linear solve handles exactly). The result is concatenated
    to (N, 4).

    Args:
        images: list of N real images (uint8 from step 3).
        masks:  optional list of N boolean valid-pixel masks (from np.isfinite of the
                ORIGINAL inputs).
        subpixel, upsampleFactor: forwarded to the pairwise estimators.
        highPass, nRho, nPhi, rmin: forwarded to pairwiseSimilarityTransform.

    Returns params of shape (N, 4), row order = input order, columns
    (theta, log s, dy, dx). Also prints rot/scale + translation diagnostics."""
    n = len(images)

    # stage 1: rotation + scale
    pw_rs = allPairwiseSimilarityTransforms(images, masks, subpixel, upsampleFactor,
                                            highPass, nRho, nPhi, rmin)
    p_rs = solveGlobalSimilarityRotScale(pw_rs, n)
    diag_rs = checkSimilarityRotScaleResiduals(pw_rs, p_rs)

    # stage 2: de-rotate + de-scale each image, then estimate translation on the de-warp.
    # The de-warp runs on the raw inputs here (the caller passed uint8); re-derive masks
    # from np.isfinite of the de-warped float (warp out-of-source -> NaN) and re-normalize
    # to uint8 so getTranslationalShifts sees the same estimation branch as stage 1.
    dewarped_f = [similarityRotScaleImage(
                        images[i], (float(p_rs[i, 0]), float(np.exp(p_rs[i, 1]))))
                  for i in range(n)]
    dewarped_masks = [np.isfinite(d) for d in dewarped_f]
    dewarped_u8 = normalizeImagesAmplitude(dewarped_f)

    pw_t = getTranslationalShifts.allPairwiseTranslationalShifts(
                dewarped_u8, masks=dewarped_masks, subpixel=subpixel,
                upsampleFactor=upsampleFactor)
    t = getTranslationalShifts.solveGlobalTranslationalShifts(pw_t, n)
    diag_t = getTranslationalShifts.checkTranslationalShiftResiduals(pw_t, t)

    params = np.concatenate([p_rs, t], axis=1)
    print(f"[getSimilarityTransform] rot/scale: {diag_rs['nPairs']} pairs | "
          f"resRotMax={diag_rs['residualMax_rot_rad']:.4f}rad  "
          f"resRotMean={diag_rs['residualMean_rot_rad']:.4f}rad  "
          f"resScaleMax={diag_rs['residualMax_scale_log']:.4f}  "
          f"resScaleMean={diag_rs['residualMean_scale_log']:.4f} || "
          f"translation: resMax={diag_t['residualMax_px']:.4f}px  "
          f"resMean={diag_t['residualMean_px']:.4f}px")
    return params


# --------------------------------------------------------------------------- #
# self-check: 2-node known-similarity recovery (the sign arbiter)
# --------------------------------------------------------------------------- #
def _getSimilarityTransform_selfcheck():
    """Recover a known similarity between two synthetic images.

    B = similarityRotScaleImage(A, (theta_k, s_k)) -- i.e. B is A rotated+scaled by
    G = (theta_k, s_k). To align B back to A we apply G^{-1}, so the pairwise rot/scale
    S_01 (apply to B to align to A) = (-theta_k, -log s_k) and the zero-mean-gauge
    per-image difference params[1] - params[0] must equal that. This is gauge-invariant
    and is the arbiter for the theta = d_phi*(2pi/nPhi), log s = -d_rho*drho_per_px sign
    convention. Also checks end-to-end: applying the recovered per-image similarities to
    both ORIGINAL images must co-register them (phase-correlation peak at ~0).

    The pattern is a broadband band-limited texture (energy across all spectral radii --
    Fourier-Mellin needs broadband radial structure to detect a scale shift; a sparse
    few-sinusoid pattern does not), shifted to be non-negative so it routes through the
    real clamp+normalize estimation branch like an amplitude image.
    """
    rng = np.random.default_rng(0)
    H, W = 128, 128
    n = rng.standard_normal((H, W))
    F = np.fft.fftshift(np.fft.fft2(n))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    oy, ox = np.ogrid[:H, :W]
    r = np.sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
    keep = (r > 4.0) & (r < 0.45 * min(H, W))      # donut: suppress DC + Nyquist
    F = F * keep
    texture = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    base = (texture - texture.min() + 0.5).astype(np.float32)   # non-negative amplitude-like

    theta_k = 0.10        # ~5.7 deg
    s_k = 1.06
    B = similarityRotScaleImage(base, (theta_k, s_k), markInvalid=True)

    # estimation branch: clamp + normalize to uint8 (mirrors coSimilarityTransformImages)
    u8 = normalizeImagesAmplitude([clampImageAmplitude(base), clampImageAmplitude(B)])
    params = getSimilarityTransform(u8, subpixel=True)

    # gauge-invariant rot/scale check: params[1]-params[0] == (-theta_k, -log s_k)
    d_theta = params[1, 0] - params[0, 0]
    d_logs = params[1, 1] - params[0, 1]
    rot_ok = np.isclose(d_theta, -theta_k, atol=0.03)
    scale_ok = np.isclose(d_logs, -np.log(s_k), atol=0.03)

    # end-to-end: applying the recovered similarities to the ORIGINALS must co-register.
    from .similarityTransformImage import similarityTransformImages
    transformed = similarityTransformImages([base, B], params)
    t0, t1 = transformed[0], transformed[1]
    corrMap, _ = _phaseCorrelationMap(
        t0.astype(np.float32), t1.astype(np.float32),
        np.isfinite(t0), np.isfinite(t1))
    pk = np.unravel_index(int(np.nanargmax(corrMap)), corrMap.shape)
    dy_pk, dx_pk = _wraparoundPick(np.array(pk, dtype=np.float64), corrMap.shape)
    align_ok = abs(dy_pk) < 2.5 and abs(dx_pk) < 2.5

    ok = bool(rot_ok and scale_ok and align_ok)
    print(f"getSimilarityTransform self-check: {'PASS' if ok else 'FAIL'} "
          f"(rot: d_theta={d_theta:+.4f} vs {-theta_k:+.4f} ok={rot_ok}, "
          f"scale: d_logs={d_logs:+.4f} vs {-np.log(s_k):+.4f} ok={scale_ok}, "
          f"end-to-end align shift=({dy_pk:+.2f},{dx_pk:+.2f}) ok={align_ok})")
    return ok


if __name__ == "__main__":
    _getSimilarityTransform_selfcheck()