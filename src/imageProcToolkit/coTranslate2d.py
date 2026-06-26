import numpy as np
from . import getTranslationalShifts
from .fftTranslate2d import fftTranslate2d
from .clamp import clamp
from . import normalizeArray as norm

'''
    Orchestrator for the multi-image co-registration pipeline

    Co-register N images that are **already rotation-aligned** (rotation alignment is
    performed outside this toolkit) by running steps 2-3-4-5:

        step 2  toIntensity + clamp      (per `arrayScale`: complex -> |z|^2, real
                                              amplitude -> x^2, real intensity -> x;
                                              then 10*log10 intensity-dB dynamic-range
                                              clamp)
        step 3  normalizeArray           (clamped intensity -> per-image uint8)
        step 4  getTranslationalShifts   (all-pairwise phase correlation on the
                                              uint8 -> zero-mean-gauge per-image
                                              shifts (N, 2))
        step 5  fftTranslateImages       (apply each per-image shift to the
                                              **original input** via FFT phase-ramp +
                                              pad-and-crop, unit-preserving)

    The inputs may be **complex or real**, and their unit is declared by the required
    `arrayScale` argument (``'amplitude'`` or ``'intensity'``). The uint8 branch
    (steps 2-3) is estimation-only: phase correlation is brightness/scale-invariant, so
    the shifts are estimated on the uint8 and applied (step 5) to the **original
    inputs** -- the "key branch": complex in -> complex out (phase preserved for
    interferometry), real in -> real out (amplitude or intensity, as declared). This is
    why accepting complex input is useful at all; the clamp does not discard phase from
    the output.

    The estimation branch works in intensity: `arrayScale` is resolved to a real
    intensity array (`_toIntensity`) before the unit-unaware intensity clamp. The
    application branch (step 5) translates the **original** input and preserves its
    declared unit -- `arrayScale` is forwarded to `fftTranslate2d` as the input-unit
    contract (translation is linear and unit-preserving, so no conversion happens
    there).

    Masks: complex/real inputs may have NaN borders (e.g. from a prior rotation step),
    and `normalizeToUint8` maps NaN -> 0 sentinel (so the border is undetectable in the
    uint8). The valid-pixel mask is therefore derived from `np.isfinite` of the
    **original input** (before clamp), not from the uint8.

    Preconditions (asserted, not run): the inputs are already rotation-aligned.
    Steps 2-3 are run internally, so the caller passes raw complex/real images -- just
    the N arrays and the `arrayScale`.

    The orchestrator writes nothing to disk; the __main__ demo loads N .npy, runs
    2->3->4->5, prints the shifts + residuals, and shows a before/after red-cyan
    anaglyph per pair (BEFORE = input pre-shift, AFTER = step-5 co-registered,
    expecting fringes -> gray).

    Note on the real-input sub-pixel caveat (from step 5): fftTranslate2d's
    real-input branch takes np.abs of the phase-ramp IFFT, which is only faithful for
    integer shifts; sub-pixel translations of a *real* input have minor distortion from
    the np.abs fold. The complex branch (the interferometry case) has no such fold --
    the phase ramp is applied directly to the complex spectrum.
'''


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


# --------------------------------------------------------------------------- #
# batch translation (step 5)
# --------------------------------------------------------------------------- #
def fftTranslateImages(images, shifts, arrayScale, markInvalid=True):
    """Translate each image by its per-image shift (batched step 5). `shifts` is the
    (N, 2) array returned by getTranslationalShifts (columns (dy, dx), row order = input order).
    Delegates to the atomic fftTranslate2d (imageProcToolkit/fftTranslate2d.py),
    which holds the phase-ramp + pad-and-crop core; this is just the per-image loop,
    kept here in the orchestrator module (fftTranslate2d.py is atomic-only). `arrayScale`
    is the input-unit contract forwarded to fftTranslate2d (translation is
    unit-preserving). markInvalid=True (default) NaN-fills the shifted-out border --
    both complex and real hold NaN."""
    if arrayScale not in ('amplitude', 'intensity'):
        raise ValueError(f"arrayScale must be 'amplitude' or 'intensity', got {arrayScale!r}")
    images = list(images)
    if len(images) != shifts.shape[0]:
        raise ValueError(
            f"{len(images)} images but shifts has {shifts.shape[0]} rows")
    return [fftTranslate2d(img, (float(shifts[i, 0]), float(shifts[i, 1])), arrayScale,
                           markInvalid)
            for i, img in enumerate(images)]


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def coTranslate2d(images, arrayScale, masks=None):
    """Co-register N already-rotation-aligned complex-or-real images (steps 2-3-4-5).

    Args:
        images: list of N complex or real images, already rotation-aligned
                (rotation alignment is performed outside this toolkit). May contain NaN borders; these are masked for shift
                estimation (derived from the input, not the uint8).
        arrayScale: required, the unit of the input images -- ``'amplitude'`` or
                ``'intensity'``. Drives the step-2 conversion to intensity
                (complex -> |z|^2, real amplitude -> x^2, real intensity -> passthrough)
                and is forwarded to step 5 as the input-unit contract (translation is
                unit-preserving).
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
        diag:       checkTranslationalShiftResiduals dict (residualMax_px, residualMean_px,
                    nPairs) -- zero-mean gauge: sum(shifts) = 0, no ground-truth
                    image.
    """
    if arrayScale not in ('amplitude', 'intensity'):
        raise ValueError(f"arrayScale must be 'amplitude' or 'intensity', got {arrayScale!r}")
    images = [np.asarray(img) for img in images]

    if masks is None:
        masks = [np.isfinite(img) for img in images]

    # steps 2 & 3: resolve to intensity per arrayScale -> 10*log10 intensity-dB clamp
    # -> per-image uint8. This is the estimation branch; it does NOT feed step 5.
    clamped = [clamp(_toIntensity(img, arrayScale)) for img in images]
    u8 = [norm.normalizeToUint8(a) for a in np.log10(clamped)]

    # step 4: all-pairwise phase-correlation shifts on the uint8 (with the
    # input-derived masks) -> zero-mean-gauge per-image shifts. Call the components
    # (not getTranslationalShifts.getTranslationalShifts) so the diag dict is returned alongside t.
    pw = getTranslationalShifts.allPairwiseTranslationalShifts(u8, masks=masks)
    t = getTranslationalShifts.solveGlobalTranslationalShifts(pw, len(images))
    diag = getTranslationalShifts.checkTranslationalShiftResiduals(pw, t)

    # step 5: apply each per-image shift to the **original input** (not the uint8 /
    # not the clamped intensity) via FFT phase-ramp + pad-and-crop. This is the key
    # branch: complex in -> complex out (phase preserved), real in -> real out.
    # arrayScale is the input-unit contract (translation is unit-preserving).
    # markInvalid=True (default) NaN-fills the shifted-out border.
    translated = fftTranslateImages(images, t, arrayScale)

    return translated, t, diag