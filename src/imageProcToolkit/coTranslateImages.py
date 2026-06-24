import numpy as np
from . import getShifts
from .fftTranslateImage import fftTranslateImage
from . import clampImageAmplitude as clamp
from . import normalizeImageAmplitude as norm

'''
    Orchestrator for the multi-image co-registration pipeline

    Co-register N images that are **already rotation-aligned** (step 1
    `groundPlaneMatching` applied externally) by running steps 2-3-4-5:

        step 2  clampImageAmplitude          (complex -> amplitude, or real passthrough;
                                              dB-histogram-mode dynamic-range clamp)
        step 3  normalizeImageAmplitude       (clamped amplitude -> per-image uint8)
        step 4  getShifts                     (all-pairwise phase correlation on the
                                              uint8 -> zero-mean-gauge per-image
                                              shifts (N, 2))
        step 5  fftTranslateImages             (apply each per-image shift to the
                                              **original input** via FFT phase-ramp +
                                              pad-and-crop)

    The inputs may be **complex or real**. The uint8 branch (steps 2-3) is
    estimation-only: phase correlation is brightness/scale-invariant, so the shifts
    are estimated on the uint8 and applied (step 5) to the **original inputs** -- the
    "key branch": complex in -> complex out (phase preserved for interferometry),
    real in -> real out (amplitude). This is why accepting complex input is useful at
    all; the clamp does not discard phase from the output.

    Masks: complex/real inputs may have NaN borders (e.g. step-1 rotated footprints),
    and `normalizeToUint8` maps NaN -> 0 sentinel (so the border is undetectable in the
    uint8). The valid-pixel mask is therefore derived from `np.isfinite` of the
    **original input** (before clamp), not from the uint8.

    Preconditions (asserted, not run): the inputs are already rotation-aligned (step 1
    external). Steps 2-3 are run internally, so the caller passes raw complex/real L2
    images -- no params.json, no shared groundPlaneMetaData.json, just the N arrays.

    The orchestrator writes nothing to disk; the __main__ demo loads N .npy, runs
    2->3->4->5, prints the shifts + residuals, and shows a before/after red-cyan
    anaglyph per pair (BEFORE = input pre-shift, AFTER = step-5 co-registered,
    expecting fringes -> gray).

    Note on the real-input sub-pixel caveat (from step 5): fftTranslateImage's
    real-input branch takes np.abs of the phase-ramp IFFT, which is only faithful for
    integer shifts; sub-pixel translations of a *real* input have minor distortion from
    the np.abs fold. The complex branch (the interferometry case) has no such fold --
    the phase ramp is applied directly to the complex spectrum.
'''


# --------------------------------------------------------------------------- #
# batch translation (step 5)
# --------------------------------------------------------------------------- #
def fftTranslateImages(images, shifts, markInvalid=True):
    """Translate each image by its per-image shift (batched step 5). `shifts` is the
    (N, 2) array returned by getShifts (columns (dy, dx), row order = input order).
    Delegates to the atomic fftTranslateImage (imageProcToolkit/fftTranslateImage.py),
    which holds the phase-ramp + pad-and-crop core; this is just the per-image loop,
    kept here in the orchestrator module (fftTranslateImage.py is atomic-only). markInvalid=True
    (default) NaN-fills the shifted-out border -- both complex and real hold NaN."""
    images = list(images)
    if len(images) != shifts.shape[0]:
        raise ValueError(
            f"{len(images)} images but shifts has {shifts.shape[0]} rows")
    return [fftTranslateImage(img, (float(shifts[i, 0]), float(shifts[i, 1])), markInvalid)
            for i, img in enumerate(images)]


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def coTranslateImages(images, masks=None):
    """Co-register N already-rotation-aligned complex-or-real images (steps 2-3-4-5).

    Args:
        images: list of N complex or real images, already rotation-aligned (step 1
                external). May contain NaN borders; these are masked for shift
                estimation (derived from the input, not the uint8).
        masks:  optional list of N boolean valid-pixel masks. If None, derived from
                np.isfinite of each input. NB: masks must come from the input, NOT the
                uint8 -- normalizeToUint8 maps NaN -> 0 sentinel, so NaN is
                undetectable in the uint8 (see step 3).

    Returns (translated, shifts, diag):
        translated: list of N images, each the **original input** shifted by its
                    per-image (dy, dx) via fftTranslateImages (markInvalid=True ->
                    NaN border). dtype is preserved: complex in -> complex out
                    (phase preserved), real in -> real out.
        shifts:     (N, 2) array, columns (dy, dx), row order = input order.
        diag:       checkShiftResiduals dict (residualMax_px, residualMean_px,
                    nPairs) -- zero-mean gauge: sum(shifts) = 0, no ground-truth
                    image.
    """
    images = [np.asarray(img) for img in images]

    if masks is None:
        masks = [np.isfinite(img) for img in images]

    # steps 2 & 3: clamp (complex -> amplitude, or real passthrough) -> per-image
    # uint8. This is the estimation branch; it does NOT feed step 5.
    clamped = [clamp.clampImageAmplitude(img) for img in images]
    u8 = norm.normalizeImagesAmplitude(clamped)

    # step 4: all-pairwise phase-correlation shifts on the uint8 (with the
    # input-derived masks) -> zero-mean-gauge per-image shifts. Call the components
    # (not getShifts.getShifts) so the diag dict is returned alongside t.
    pw = getShifts.allPairwiseShifts(u8, masks=masks)
    t = getShifts.solveGlobalShifts(pw, len(images))
    diag = getShifts.checkShiftResiduals(pw, t)

    # step 5: apply each per-image shift to the **original input** (not the uint8 /
    # not the clamped amplitude) via FFT phase-ramp + pad-and-crop. This is the key
    # branch: complex in -> complex out (phase preserved), real in -> real out.
    # markInvalid=True (default) NaN-fills the shifted-out border.
    translated = fftTranslateImages(images, t)

    return translated, t, diag