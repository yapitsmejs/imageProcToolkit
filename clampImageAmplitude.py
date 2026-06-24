import numpy as np

'''
    Convert a complex image to amplitude (no-op if already real) and clamp its
    dynamic range to a band ideal for correlation-based registration / display:
    suppress the bright-scatterer tail that would otherwise dominate the phase
    spectrum (step 4) and saturate any min-max normalization (step 3 / anaglyph).
    Works on both complex and real input -- complex is reduced to amplitude (|.|)
    first (phase discarded, since the clamp operates on brightness only); real
    input is passed through as float32.

    This is the dB-histogram-mode clamp from the tested reference at
    temp/_registration.py (reduceDynamicRange), refactored so that:

      * it returns the clamped *linear amplitude* (not a normalized image) --
        normalization is decoupled into imageProcToolkit/normalizeImageAmplitude.py (step 3);
      * the input ndim is preserved (the reference added a channel axis at line 77
        and never squeezed it);
      * nanmax / |crossPower| divisions and the all-NaN / all-zero cases are guarded
        (the reference divided by np.nanmax unguarded at line 79);
      * it is pure numpy -- the reference's cv2.normalize (only used in the normalize
        half, not the clamp half) is not needed here.

    Radio calibration is out of scope (same as the rest of the pipeline).
'''


def toAmplitude(image):
    """Complex -> amplitude (|.|, float32); real input passed through as float32."""
    if np.iscomplexobj(image):
        return np.abs(image).astype(np.float32)
    return np.asarray(image, dtype=np.float32)


def amplitudePowerDB(amp):
    """Power in dB relative to the image's own max: 20*log10(amp / nanmax(amp)).

    Range is (-inf, 0] with 0 at the brightest pixel. NaNs propagate. Returns an
    all-NaN array if the input is all-NaN or all-<=0 (no valid reference max). Exact-zero
    valid pixels (common in amplitude images -- noise floor / zero-fill) map to -inf,
    which the clamp's later `clip(dB, lo, hi)` pulls up to the lower edge (the correct
    behaviour); the log10(0) that produces them is silenced here so it does not raise a
    spurious RuntimeWarning."""
    amp = np.asarray(amp, dtype=np.float32)
    mx = np.nanmax(amp)
    if not np.isfinite(mx) or mx <= 0:
        return np.full_like(amp, np.nan, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 20.0 * np.log10(amp / mx)


def dBHistogramMode(dBimage, outputDynamicRangePowerDB=60.0,
                    inputDynamicRangePowerDB=None, binsPerDB=5):
    """Mode of the dB-amplitude histogram, smoothed by a moving-average window of
    width outputDynamicRangePowerDB (in dB). This is the reference's mode finder
    (lines 82-92): the bulk of a SAR amplitude histogram sits in a narrow dB band
    (the clutter mode), well below the bright-scatterer tail, so the mode -- not the
    mean/median -- is the right centre for the clamp window.

    Args:
        dBimage: dB amplitude (from amplitudePowerDB). Non-finite values are ignored.
        outputDynamicRangePowerDB: desired output dynamic range in dB (window width).
        inputDynamicRangePowerDB: histogram lower bound in dB (positive). If None,
            inferred from the most-negative finite dB value.
        binsPerDB: histogram resolution.

    Returns:
        float: the dB bin centre of the smoothed histogram maximum (the mode).
    """
    dBimage = np.asarray(dBimage, dtype=np.float32)
    finite = dBimage[np.isfinite(dBimage)]
    if finite.size == 0:
        raise ValueError("dB image has no finite values")
    if inputDynamicRangePowerDB is None:
        inputDynamicRangePowerDB = float(np.abs(np.nanmin(finite)))
    if not np.isfinite(inputDynamicRangePowerDB) or inputDynamicRangePowerDB <= 0:
        inputDynamicRangePowerDB = 1.0

    histBins = max(1, int(inputDynamicRangePowerDB * binsPerDB))
    hist, edges = np.histogram(finite, bins=histBins,
                               range=(-inputDynamicRangePowerDB, 0.0), density=True)
    binCenters = 0.5 * (edges[:-1] + edges[1:])
    if binCenters.size < 2:
        return float(binCenters[0])
    binWidth = (binCenters[-1] - binCenters[0]) / (binCenters.size - 1)
    windowSize = max(1, int(round(outputDynamicRangePowerDB / binWidth)))
    window = np.ones(windowSize, dtype=np.float64) / windowSize
    smoothed = np.convolve(hist, window, mode='same')
    return float(binCenters[int(np.argmax(smoothed))])


def clampAmplitude(amp, outputDynamicRangePowerDB=60.0,
                   inputDynamicRangePowerDB=None, binsPerDB=5):
    """Clamp amplitude dynamic range to +/-outputDynamicRangePowerDB/2 around the dB
    histogram mode. Returns the clamped *linear amplitude* (same shape/dtype as the
    input amplitude). NaNs are preserved; the input ndim is preserved.

    Bright scatterers above the window are pulled down to the upper edge and the
    noise floor below the window is pulled up to the lower edge, so a subsequent
    min-max normalization (step 3) spreads the bulk of the scene across the full
    [0, 255] range instead of being crushed to black by a few bright points."""
    amp = np.asarray(amp, dtype=np.float32)
    dB = amplitudePowerDB(amp)
    if not np.isfinite(np.nanmax(dB)):
        return amp.copy()
    mode = dBHistogramMode(dB, outputDynamicRangePowerDB,
                           inputDynamicRangePowerDB, binsPerDB)
    lo = mode - outputDynamicRangePowerDB / 2.0
    hi = mode + outputDynamicRangePowerDB / 2.0
    clipped = np.clip(dB, lo, hi)          # NaN -> NaN (min/max propagate NaN)
    return (10.0 ** (clipped / 20.0)).astype(np.float32)


def clampImageAmplitude(image, outputDynamicRangePowerDB=60.0,
                        inputDynamicRangePowerDB=None):
    """Complex (or real) image -> clamped linear amplitude.

    Convenience wrapper: toAmplitude then clampAmplitude. Accepts complex or real
    input -- complex is reduced to amplitude (phase discarded), real is passed
    through. Returns float32 amplitude with the dynamic range clamped; does NOT
    normalize (see step 3)."""
    amp = toAmplitude(image)
    return clampAmplitude(amp, outputDynamicRangePowerDB, inputDynamicRangePowerDB)