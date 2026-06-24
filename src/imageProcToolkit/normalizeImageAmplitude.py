import numpy as np

'''
    Step 3 of the multi-image co-registration pipeline (see REGISTRATION_PLAN.md).

    Per-image min-max map to uint8 [0, 255] for the estimation / display branch. Works
    on both complex and real input -- complex is reduced to amplitude (|.|) first (the
    magnitude carries the brightness information; phase is irrelevant to a min-max
    brightness scale), real input is used directly.

    NaNs / non-finite values map to the sentinel 0. NB: because NaN -> 0, a NaN border
    is *undetectable* in the uint8 output, so a downstream step that needs a valid-pixel
    mask (e.g. step 4 phase correlation) must derive it from the **original input**'s
    finiteness, not from the uint8 (see REGISTRATION_PLAN.md).

    Radio calibration is out of scope (same as the rest of the pipeline).
'''


def normalizeToUint8(image):
    """Per-image min-max map of amplitude to uint8 [0, 255].

    Accepts complex or real input: complex is reduced to amplitude (|.|) first (phase
    discarded), real is used directly. NaNs / non-finite values map to the sentinel 0.
    Returns (uint8 image, vmin, vmax) so a caller can invert or reuse the scale.

    Args:
        image: complex or real image (complex -> amplitude first).

    Returns:
        (out, vmin, vmax): out is uint8, same shape as the input; vmin/vmax are the
        finite min/max used for the scale (0.0 and 1.0 if the input has no finite values).
    """
    image = np.asarray(image)
    if np.iscomplexobj(image):
        image = np.abs(image)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8), 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        return np.zeros(image.shape, dtype=np.uint8), vmin, vmax
    scaled = (image - vmin) / (vmax - vmin) * 255.0
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)        # NaN -> 0 sentinel
    out = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return out, vmin, vmax


def normalizeImagesAmplitude(images):
    """Normalize a list of complex-or-real images to uint8, per-image. Returns the list
    of uint8 arrays (scales are discarded -- this is the estimation/display path)."""
    return [normalizeToUint8(a)[0] for a in images]