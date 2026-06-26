import numpy as np

'''
    Step 3 of the multi-image co-registration pipeline.

    Per-image min-max map to uint8 [0, 255] for the estimation / display branch. Works
    on both complex and real input -- complex is reduced to amplitude (|.|) first (the
    magnitude carries the brightness information; phase is irrelevant to a min-max
    brightness scale), real input is used directly.

    NaNs / non-finite values map to the sentinel 0. NB: because NaN -> 0, a NaN border
    is *undetectable* in the uint8 output, so a downstream step that needs a valid-pixel
    mask (e.g. step 4 phase correlation) must derive it from the **original input**'s
    finiteness, not from the uint8.

    Radio calibration is out of scope.
'''


def normalizeToUint8(image):
    """Per-image min-max map of amplitude to uint8 [0, 255].

    Accepts complex or real input: complex is reduced to amplitude (|.|) first (phase
    discarded), real is used directly. NaNs / non-finite values map to the sentinel 0.
    Returns the uint8 image only -- this is the estimation / display path, so the scale
    is discarded (no inversion).

    Args:
        image: complex or real image (complex -> amplitude first).

    Returns:
        out: uint8 image, same shape as the input. If the input has no finite values,
        or its finite range is degenerate (vmax <= vmin), a zero uint8 array is returned.
    """
    image = np.asarray(image)
    if np.iscomplexobj(image):
        image = np.abs(image)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - vmin) / (vmax - vmin) * 255.0
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)        # NaN -> 0 sentinel
    out = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return out