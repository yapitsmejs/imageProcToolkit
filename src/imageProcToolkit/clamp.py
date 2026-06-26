import numpy as np

'''
    Clamp an n-dimensional *intensity* array's dynamic range to a band ideal for
    correlation-based registration / display: suppress the bright-scatterer tail
    that would otherwise dominate the phase spectrum (step 4) and saturate any
    min-max normalization (step 3 / anaglyph).

    This is an *intensity*-domain, unit-unaware clamp. It always works in
    intensity dB via 10*log10:

        dB = 10 * log10(intensity / nanmax(intensity))        (forward)
        intensity = 10 ** (dB / 10)                            (inverse)

    The unit-unawareness is deliberate. The amplitude-dB formula 20*log10(amp) is
    *the same scale* as the intensity-dB formula, because

        20*log10(amp) == 10*log10(amp**2) == 10*log10(intensity)

    -- i.e. 20-log of an amplitude is just 10-log of the intensity viewed through
    a square. Rather than carry a 20-log amplitude path alongside a 10-log
    intensity path, this module standardizes on the single 10*log10 convention
    and works in intensity. `clamp` itself is unit-unaware: it simply operates on
    an n-dimensional *real-valued intensity* array. The amplitude -> intensity
    conversion is the *caller's* responsibility -- the orchestrators take an
    `arrayScale` and square amplitude into intensity (complex -> |z|**2, real
    amplitude -> x**2, real intensity -> passthrough) before calling `clamp`.
    `clamp` does not inspect the input's dimensionality; it only enforces that the
    output shape matches the input shape. The input is assumed to be real, finite
    non-negative intensity (NaNs propagate); a complex or un-converted amplitude
    array handed straight to `clamp` is a caller error.

    This is a dB-histogram-mode clamp, factored so that:

      * it returns the clamped *linear intensity* (not a normalized array) --
        normalization is decoupled into imageProcToolkit/normalizeArray.py (step 3);
      * the input ndim is preserved (no spurious channel axis is added);
      * nanmax / |crossPower| divisions and the all-NaN / all-zero cases are guarded;
      * it is pure numpy.

    Multi-dimensional / multi-channel input: every quantity is reduced over the
    *whole* array -- the 0-dB reference is the global `np.nanmax`, the histogram
    mode is taken over the flattened finite values, and the clamp window is
    applied elementwise. A 2D array is therefore clamped unchanged; an N-D stack
    (e.g. an `(H, W, C)` multi-channel array) is clamped *jointly*: all channels
    share one 0-dB reference, one histogram mode, and one `[mode +/- dr/2]`
    window. Joint clamping preserves the relative brightness between channels (a
    per-channel clamp would re-centre each channel's clutter on its own max and
    destroy that balance). Shape and dtype are always preserved; NaNs propagate.
    Callers that want per-slice clamping of a stack
    should clamp each 2D slice separately.

    Radio calibration is out of scope.
'''


def intensityPowerDB(intensity):
    """Intensity in dB relative to the array's global max: 10*log10(intensity / nanmax(intensity)).

    The reference `nanmax` is taken over the *whole* array, so for an N-D stack every
    channel is expressed in dB relative to one shared brightest pixel -- this is what
    makes downstream clamping *joint* across channels. Shape is preserved.

    Range is (-inf, 0] with 0 at the brightest pixel. NaNs propagate. Returns an
    all-NaN array if the input is all-NaN or all-<=0 (no valid reference max). Exact-zero
    valid pixels (noise floor / zero-fill) map to -inf, which the clamp's later
    `clip(dB, lo, hi)` pulls up to the lower edge (the correct behaviour); the log10(0)
    that produces them is silenced here so it does not raise a spurious RuntimeWarning."""
    intensity = np.asarray(intensity, dtype=np.float32)
    mx = np.nanmax(intensity)
    if not np.isfinite(mx) or mx <= 0:
        return np.full_like(intensity, np.nan, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(intensity / mx)


def dBHistogramMode(dBarray, outputDynamicRangePowerDB=60.0,
                    inputDynamicRangePowerDB=None, binsPerDB=5):
    """Mode of the dB histogram, smoothed by a moving-average window of
    width outputDynamicRangePowerDB (in dB): the bulk of a SAR intensity histogram
    sits in a narrow dB band (the clutter mode), well below the bright-scatterer
    tail, so the mode -- not the mean/median -- is the right centre for the clamp
    window.

    Args:
        dBarray: intensity dB (from intensityPowerDB). Non-finite values are ignored.
        outputDynamicRangePowerDB: desired output dynamic range in dB (window width).
        inputDynamicRangePowerDB: histogram lower bound in dB (positive). If None,
            inferred from the most-negative finite dB value.
        binsPerDB: histogram resolution.

    Returns:
        float: the dB bin centre of the smoothed histogram maximum (the mode).
    """
    dBarray = np.asarray(dBarray, dtype=np.float32)
    finite = dBarray[np.isfinite(dBarray)]
    if finite.size == 0:
        raise ValueError("dB array has no finite values")
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


def _clampCore(intensity, outputDynamicRangePowerDB=60.0,
               inputDynamicRangePowerDB=None, binsPerDB=5):
    """Clamp intensity dynamic range to +/-outputDynamicRangePowerDB/2 around the dB
    histogram mode. Returns the clamped *linear intensity* (same shape/dtype as the
    input intensity). NaNs are preserved; the input ndim is preserved.

    Internal core operating on an intensity array; the public entry point is
    `clamp` (intensity array -> this core).

    The dB reference (global `nanmax`), the histogram mode (over all finite values),
    and the clamp window are all computed over the whole array, so an N-D / stacked
    input is clamped *jointly* across all channels with one shared window (see the
    module docstring). For per-slice clamping of a stack, call this once per 2D slice.

    Bright scatterers above the window are pulled down to the upper edge and the
    noise floor below the window is pulled up to the lower edge, so a subsequent
    min-max normalization (step 3) spreads the bulk of the scene across the full
    [0, 255] range instead of being crushed to black by a few bright points."""
    intensity = np.asarray(intensity, dtype=np.float32)
    dB = intensityPowerDB(intensity)
    if not np.isfinite(np.nanmax(dB)):
        return intensity.copy()
    mode = dBHistogramMode(dB, outputDynamicRangePowerDB,
                           inputDynamicRangePowerDB, binsPerDB)
    lo = mode - outputDynamicRangePowerDB / 2.0
    hi = mode + outputDynamicRangePowerDB / 2.0
    clipped = np.clip(dB, lo, hi)          # NaN -> NaN (min/max propagate NaN)
    return (10.0 ** (clipped / 10.0)).astype(np.float32)


def clamp(array, outputDynamicRangePowerDB=60.0,
          inputDynamicRangePowerDB=None):
    """N-D intensity array -> clamped linear intensity array (same shape).

    Operates on an n-dimensional *intensity* array. The 0-dB reference (global
    `nanmax`), the dB-histogram mode, and the `[mode +/- dr/2]` clamp window are all
    taken over the whole array, so an N-D stack is clamped *jointly* across all axes
    with one shared window (preserving the relative brightness between slices); a 2D
    array is clamped unchanged. The input is assumed to be real-valued intensity
    already -- convert complex / amplitude to intensity first (the orchestrators do
    this via their `arrayScale` argument: complex -> |z|**2, real amplitude -> x**2,
    real intensity -> passthrough). The output is float32 clamped linear intensity
    and is NOT normalized (see step 3). For per-slice clamping of a stack, call this
    once per 2D slice.

    No dimensionality is assumed or checked -- the clamp is elementwise and works
    for any ndim; the only invariant enforced is that the output shape matches the
    input shape."""
    out = _clampCore(array, outputDynamicRangePowerDB, inputDynamicRangePowerDB)
    assert out.shape == np.asarray(array).shape
    return out


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _selfCheck():
    """Synthetic self-check (no data needed) that the clamp is joint over the whole
    array for N-D input and unchanged for 2D.

    Asserts:
      (1) 2D: shape/dtype preserved, output finite within the expected dB window.
      (2) joint == flatten-then-reshape: clamp on an (H, W, 3) intensity stack is
          bit-identical to clamping the same data flattened to 1D and reshaped (same
          global max, same flattened histogram -> same mode -> same clip).
      (3) joint != independent: clamping the (H, W, 3) stack jointly differs from
          clamping each 2D slice separately when the slices have different scales.
      (4) NaN: an all-NaN channel stays NaN; a mixed stack clamps the finite slices
          jointly and leaves NaN as NaN.
    """
    rng = np.random.default_rng(0)

    # (1) 2D: real non-negative intensity (squared normal -> |z|^2-like).
    img2d = (rng.standard_normal((32, 32)) ** 2).astype(np.float32)
    out2d = clamp(img2d)
    c2 = (out2d.shape == img2d.shape and out2d.dtype == np.float32
          and np.all(np.isfinite(out2d)))

    # (2) joint == flatten-then-reshape on an (H, W, 3) intensity stack with channels
    # of different scales (one bright, one mid, one dim).
    base = (rng.standard_normal((24, 24)) ** 2).astype(np.float32)
    stack = np.stack([base, 0.1 * base, 0.01 * base], axis=-1)   # (24, 24, 3)
    joint = clamp(stack)
    flat = clamp(stack.reshape(-1, order='C')).reshape(stack.shape, order='C')
    eq_flat = np.array_equal(joint, flat, equal_nan=True)

    # (3) joint != independent per-slice
    indep = np.stack([clamp(stack[..., c]) for c in range(3)], axis=-1)
    neq_indep = not np.array_equal(joint, indep, equal_nan=True)

    # (4) NaN: all-NaN channel + a mixed stack
    nanstack = stack.copy()
    nanstack[..., 0] = np.nan
    out_nan = clamp(nanstack)
    nan_ch0_allnan = bool(np.all(np.isnan(out_nan[..., 0])))
    nan_finite_ok = bool(np.all(np.isfinite(out_nan[..., 1:]))
                         and out_nan[..., 1:].shape == (24, 24, 2))

    ok = c2 and eq_flat and neq_indep and nan_ch0_allnan and nan_finite_ok
    print("--- clamp self-check ---")
    print(f"  (1) 2D shape/dtype/finite           : {'PASS' if c2 else 'FAIL'}")
    print(f"  (2) joint == flatten-then-reshape   : {'PASS' if eq_flat else 'FAIL'}")
    print(f"  (3) joint != independent per-slice  : {'PASS' if neq_indep else 'FAIL'}")
    print(f"  (4) NaN channel preserved + finite  : "
          f"{'PASS' if (nan_ch0_allnan and nan_finite_ok) else 'FAIL'}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selfCheck()