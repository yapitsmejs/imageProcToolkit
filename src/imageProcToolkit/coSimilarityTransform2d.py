import numpy as np
from . import getSimilarityTransform
from . import getTranslationalShifts
from . import similarityTransform2d
from .clamp import clamp
from . import normalizeArray as norm


def _toIntensity(array, arrayScale):
    """Resolve an input array to a real float32 intensity array per `arrayScale`.

    complex -> |z|**2 (intensity, phase discarded); real + ``'amplitude'`` -> x**2
    (square the amplitude into intensity); real + ``'intensity'`` -> x (already
    intensity, passthrough). This is the amplitude -> intensity conversion the
    orchestrator owes the unit-unaware `clamp` (which only speaks intensity)."""
    if np.iscomplexobj(array):
        a = np.abs(array).astype(np.float32)
        return a * a
    x = np.asarray(array, dtype=np.float32)
    return x * x if arrayScale == 'amplitude' else x

'''
    Orchestrator for multi-image co-registration under a 4-DOF similarity per image
    (rotation theta + uniform scale s + translation (dy, dx)), generalizing
    coTranslate2d from the 2-DOF translation group to the 4-DOF similarity group.

    Co-register N images (which need **not** be rotation-aligned -- the similarity
    estimator recovers rotation; if they are already aligned, theta_i ~ 0 and s_i ~ 1
    and this reduces to coTranslate2d) by running steps 2-3-4-5:

        step 2  toIntensity + clamp      (per `arrayScale`: complex -> |z|^2, real
                                              amplitude -> x^2, real intensity -> x;
                                              then 10*log10 intensity-dB dynamic-range
                                              clamp)
        step 3  normalizeArray           (clamped intensity -> per-image uint8)
        step 4  getSimilarityTransform   (stage 1: all-pairwise Fourier-Mellin
                                              rot/scale on the uint8 magnitude spectra
                                              -> zero-mean-gauge per-image (theta, log s);
                                              stage 2: de-rotate + de-scale each clamped
                                              intensity, re-normalize, then reuse
                                              getTranslationalShifts for (dy, dx))
        step 5  similarityTransformImages (apply each per-image similarity to the
                                              **original input** via bilinear rot/scale
                                              warp + FFT phase-ramp translate,
                                              unit-preserving)

    As in coTranslate2d, the uint8 branch (steps 2-3 + the stage-1 spectra) is
    estimation-only: phase correlation / Fourier-Mellin are brightness-invariant, so the
    similarity is estimated on the uint8 / clamped intensity and applied (step 5) to the
    **original inputs** -- the "key branch": complex in -> complex out (phase preserved
    for interferometry), real in -> real out. The clamp does not discard phase from the
    output. The required `arrayScale` declares the inputs' unit; the estimation branch
    resolves it to intensity (`_toIntensity`) before the unit-unaware clamp, and the
    application branch forwards it to similarityTransformImages as the input-unit
    contract (a similarity warp is unit-preserving).

    Masks: complex/real inputs may have NaN borders, and normalizeToUint8 maps NaN -> 0
    sentinel (so the border is undetectable in the uint8). The valid-pixel mask for
    stage 1 is therefore derived from np.isfinite of the **original input** (before
    clamp), not from the uint8. For stage 2 the mask is re-derived from np.isfinite of
    the de-warped clamped intensity (the warp marks its out-of-source border NaN).

    The decoupled two-stage solve and the gauge (stage-1 zero-mean on (theta, log s),
    stage-2 zero-mean on (dy, dx); 4 gauge DOF fixed, no image is ground truth) are
    documented in getSimilarityTransform.py. The orchestrator writes nothing to disk.

    Note on the real-input sub-pixel caveat (from step 5, carried over from
    coTranslate2d): similarityTransform2d's translate stage takes np.abs of the
    phase-ramp IFFT for real input, faithful for ~integer shifts with minor distortion
    for sub-pixel real translations; the complex branch (interferometry) has no fold.
'''


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def coSimilarityTransform2d(images, arrayScale, masks=None, subpixel=True, upsampleFactor=1,
                                highPass=True, nRho=None, nPhi=None, rmin=1.0,
                                masterIndex=None):
    """Co-register N complex-or-real images under a 4-DOF similarity per image
    (steps 2-3-4-5; rotation need not be pre-aligned).

    Args:
        images: list of N complex or real images. Need not be rotation-aligned (the
                similarity estimator recovers rotation). May contain NaN borders; these
                are masked for estimation (derived from the input, not the uint8).
        arrayScale: required, the unit of the input images -- ``'amplitude'`` or
                ``'intensity'``. Drives the step-2 conversion to intensity
                (complex -> |z|^2, real amplitude -> x^2, real intensity -> passthrough)
                and is forwarded to step 5 as the input-unit contract (a similarity warp
                is unit-preserving).
        masks:  optional list of N boolean valid-pixel masks. If None, derived from
                np.isfinite of each input. NB: masks must come from the input, NOT the
                uint8 -- normalizeToUint8 maps NaN -> 0 sentinel, so NaN is undetectable
                in the uint8.
        subpixel, upsampleFactor, highPass, nRho, nPhi, rmin: forwarded to the stage-1
                pairwise Fourier-Mellin estimator (see getSimilarityTransform).
        masterIndex: None (default) for the zero-mean gauge (4 gauge DOF fixed, no image
                is ground truth -- the correction is distributed across all images), or
                an int node index (negative wraps, so -1 = the last image) to pin that
                image at the identity and register every other image toward it. The same
                master is used for both the rot/scale stage and the translation stage so
                the composed similarity pins the master end-to-end. Setting it also
                switches estimation to the O(n) star graph (only the n-1 master<->image
                pairs per stage) instead of all n(n-1)/2 pairs.

    Returns (transformed, params, diag):
        transformed: list of N images, each the **original input** warped by its per-image
                    similarity via similarityTransformImages (markInvalid=True -> NaN
                    border). dtype is preserved: complex in -> complex out (phase
                    preserved), real in -> real out.
        params:      (N, 4) array, columns (theta, log s, dy, dx), row order = input
                    order. By default the zero-mean gauge holds per component (no
                    ground-truth image); with masterIndex set, params[masterIndex] =
                    (0, 0, 0, 0). NB the scale column is log s -- exp it for the
                    applied scale (similarityTransformImages does this internally).
        diag:        {'rotScale': checkSimilarityRotScaleResiduals dict,
                      'translation': checkTranslationalShiftResiduals dict}.
    """
    if arrayScale not in ('amplitude', 'intensity'):
        raise ValueError(f"arrayScale must be 'amplitude' or 'intensity', got {arrayScale!r}")
    images = [np.asarray(img) for img in images]
    n = len(images)

    if masks is None:
        masks = [np.isfinite(img) for img in images]

    # When a master is nominated, estimate only the n-1 master<->image pairs (a star
    # graph centered on the master) in BOTH stages instead of all n(n-1)/2 pairs --
    # O(n) vs O(n^2) estimation. The same master k is used for both stages so the
    # composed similarity pins the master end-to-end. The (min, max) edge keying matches
    # the pairwise sign convention, so the fix-node solve yields per-image params
    # relative to k.
    starPairs = None
    if masterIndex is not None:
        k = masterIndex % n
        starPairs = [(min(i, k), max(i, k)) for i in range(n) if i != k]

    # steps 2 & 3: resolve to intensity per arrayScale -> 10*log10 intensity-dB clamp ->
    # per-image uint8. This is the estimation branch; it does NOT feed step 5.
    clamped = [clamp(_toIntensity(img, arrayScale)) for img in images]
    u8 = [norm.normalizeToUint8(a) for a in np.log10(clamped)]

    # stage 1: pairwise Fourier-Mellin rotation+scale on the uint8 magnitude spectra
    # (with the input-derived masks) -> per-image (theta, log s) under the chosen gauge.
    # All pairs (zero-mean gauge) when masterIndex is None, or the n-1 star pairs
    # (fix-node gauge) when set. Call the components (not
    # getSimilarityTransform.getSimilarityTransform) so the diag dict is returned
    # alongside p_rs and so stage 2 de-warps the clamped intensity (not the uint8).
    pw_rs = getSimilarityTransform.allPairwiseSimilarityTransforms(
                u8, masks=masks, subpixel=subpixel, upsampleFactor=upsampleFactor,
                highPass=highPass, nRho=nRho, nPhi=nPhi, rmin=rmin, pairs=starPairs)
    p_rs = getSimilarityTransform.solveGlobalSimilarityRotScale(pw_rs, n,
                                                                masterIndex=masterIndex)
    diag_rs = getSimilarityTransform.checkSimilarityRotScaleResiduals(pw_rs, p_rs)

    # stage 2: de-rotate + de-scale the **clamped intensity** (float32) by (theta_i, s_i),
    # re-derive masks from np.isfinite of the de-warped float (warp out-of-source -> NaN),
    # and re-normalize to uint8. The de-warp physically removes the rot/scale, leaving a
    # pure-translation residual that getTranslationalShifts handles exactly.
    dewarped_f = [similarityTransform2d.similarityRotScaleImage(
                        clamped[i], (float(p_rs[i, 0]), float(np.exp(p_rs[i, 1]))))
                  for i in range(n)]
    dewarped_masks = [np.isfinite(d) for d in dewarped_f]
    dewarped_u8 = [norm.normalizeToUint8(a) for a in dewarped_f]

    pw_t = getTranslationalShifts.allPairwiseTranslationalShifts(
                dewarped_u8, masks=dewarped_masks, subpixel=subpixel,
                upsampleFactor=upsampleFactor, pairs=starPairs)
    t = getTranslationalShifts.solveGlobalTranslationalShifts(pw_t, n, masterIndex=masterIndex)
    diag_t = getTranslationalShifts.checkTranslationalShiftResiduals(pw_t, t)

    # compose (N, 4) = (theta, log s, dy, dx) and apply to the ORIGINAL inputs (step 5):
    # complex in -> complex out (phase preserved), real in -> real out. arrayScale is the
    # input-unit contract (a similarity warp is unit-preserving).
    params = np.concatenate([p_rs, t], axis=1)
    transformed = similarityTransform2d.similarityTransformImages(images, params, arrayScale)

    diag = {'rotScale': diag_rs, 'translation': diag_t}
    print(f"[coSimilarityTransform2d] rot/scale: {diag_rs['nPairs']} pairs | "
          f"resRotMax={diag_rs['residualMax_rot_rad']:.4f}rad  "
          f"resScaleMax={diag_rs['residualMax_scale_log']:.4f} || "
          f"translation: resMax={diag_t['residualMax_px']:.4f}px  "
          f"resMean={diag_t['residualMean_px']:.4f}px")
    return transformed, params, diag


# --------------------------------------------------------------------------- #
# self-check: 2-image full-similarity end-to-end alignment
# --------------------------------------------------------------------------- #
def _coSimilarityTransform2d_selfcheck():
    """End-to-end: co-register two images related by a known full similarity
    (rotation + scale + translation) and confirm the result aligns.

    B = similarityTransform2d(A, (theta_k, s_k, dy_k, dx_k), arrayScale). coSimilarityTransform2d
    returns per-image similarities under a zero-mean gauge; applying them to both ORIGINAL
    images must co-register the pair (phase-correlation peak at ~0). Also checks the
    recovered params shape and the nested diag dict. The pattern is the same broadband
    band-limited amplitude texture used by getSimilarityTransform's self-check.
    """
    from ._phaseCorrelationCore import _phaseCorrelationMap, _wraparoundPick

    rng = np.random.default_rng(0)
    H, W = 128, 128
    n = rng.standard_normal((H, W))
    F = np.fft.fftshift(np.fft.fft2(n))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    oy, ox = np.ogrid[:H, :W]
    r = np.sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
    F = F * ((r > 4.0) & (r < 0.45 * min(H, W)))
    texture = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    base = (texture - texture.min() + 0.5).astype(np.float32)

    theta_k = 0.09
    s_k = 1.05
    dy_k, dx_k = 3.0, -2.0          # ~integer so the real-input translate is faithful
    B = similarityTransform2d.similarityTransform2d(
            base, (theta_k, s_k, dy_k, dx_k), 'amplitude', markInvalid=True)

    transformed, params, diag = coSimilarityTransform2d([base, B], 'amplitude')

    shape_ok = params.shape == (2, 4) and len(transformed) == 2
    diag_ok = ('rotScale' in diag and 'translation' in diag
               and 'residualMax_rot_rad' in diag['rotScale']
               and 'residualMax_px' in diag['translation'])

    # end-to-end alignment: the two transformed ORIGINAL images must co-register.
    t0, t1 = transformed[0], transformed[1]
    corrMap, _ = _phaseCorrelationMap(
        t0.astype(np.float32), t1.astype(np.float32),
        np.isfinite(t0), np.isfinite(t1))
    pk = np.unravel_index(int(np.nanargmax(corrMap)), corrMap.shape)
    dy_pk, dx_pk = _wraparoundPick(np.array(pk, dtype=np.float64), corrMap.shape)
    align_ok = abs(dy_pk) < 2.5 and abs(dx_pk) < 2.5

    # gauge-invariant rot/scale difference should match the inverse of the applied G.
    d_theta = params[1, 0] - params[0, 0]
    d_logs = params[1, 1] - params[0, 1]
    rot_ok = np.isclose(d_theta, -theta_k, atol=0.03)
    scale_ok = np.isclose(d_logs, -np.log(s_k), atol=0.03)

    ok = bool(shape_ok and diag_ok and align_ok and rot_ok and scale_ok)
    print(f"coSimilarityTransform2d self-check: {'PASS' if ok else 'FAIL'} "
          f"(shape={shape_ok}, diag={diag_ok}, "
          f"rot: {d_theta:+.4f} vs {-theta_k:+.4f} ok={rot_ok}, "
          f"scale: {d_logs:+.4f} vs {-np.log(s_k):+.4f} ok={scale_ok}, "
          f"align shift=({dy_pk:+.2f},{dx_pk:+.2f}) ok={align_ok})")
    return ok


if __name__ == "__main__":
    _coSimilarityTransform2d_selfcheck()