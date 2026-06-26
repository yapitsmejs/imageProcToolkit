import numpy as np

'''
    Shift convention

    pairwiseTranslationalShift(i, j) returns s_ij = the translation to apply to **j** to align it
    to **i** (F(i) * conj(F(j)), peak -> shift applied to j). getTranslationalShifts returns
    per-image shifts t_i of shape (N, 2), columns (dy, dx), such that the residual
    res_ij = s_ij - (t_j - t_i) is minimized; driving it to zero gives
    **t_j - t_i = s_ij**. Applying fftTranslate2d(image_i, t_i) to each image
    therefore co-registers them under the zero-mean gauge (sum t = 0, no
    ground-truth image). The reference phase ramp exp(-2j*pi*(fx*dx + fy*dy)) gives
    output[n] = input[n - (dy, dx)] -- content moves by (dy, dx) -- identical to
    getTranslationalShifts._fftShiftPeriodic, whose self-check confirms the sign.

    --------------------------------------------------------------------------- #
    Implementation notes

    * Crop-slice (zero-shift no-op): the padded array is cropped back with
      [padY:padY+H, padX:padX+W]. The naively symmetric [padY:-padY, padX:-padX]
      would collapse to the empty slice [0:0] when a pad component is 0 (zero
      shift); the explicit end-index form is a true no-op slice when pad = 0 and
      identical otherwise. Applied in both the CPU and GPU branches.
    * Dtype preservation: np.fft.fft2 upcasts complex64 -> complex128 (and float32
      -> complex128); cast back with .astype(image.dtype) at a single return point
      so the on-disk complex64 row-major contract is preserved.
    * Real-input branch: takes np.abs of the phase-ramp IFFT (the imaginary part is
      ~0 residual). Kept for parity with the GPU branch and so the self-check runs
      on a real float32 input without an awkward complex cast.
    * cupy is an optional GPU fast-path behind a guarded switch (see below) -- a
      caller that moves data to the GPU once keeps it there across chained ops
      ("move once, compute on GPU, bring back once").
    * markInvalid (default True): a pixel output[r, c] = input[r - dy, c - dx] has a
      real source sample iff 0 <= r - dy < H and 0 <= c - dx < W; pixels outside that
      region (the shifted-out border that lost data to the zero-pad) are set to NaN
      so interferometry never treats a fake zero border as real data. This grows the
      NaN border on every look -- the intended, cleaner behaviour. markInvalid=False
      leaves the shifted-out / input-NaN regions as the raw (zero-filled, possibly
      ringing) values.

    The pad-and-crop trick zero-pads by ceil(|shift|) per axis, phase-ramps the
    padded FFT, then crops back: zero-pad avoids circular wraparound but introduces
    mild edge ringing from the broken periodicity -- fine for the small registration
    shifts expected here, and strictly better than a raw circular shift.

    Radio calibration is out of scope.
'''

# cupy is an optional GPU fast-path for the phase-ramp FFTs. A usable GPU is not
# required: detect at import whether a CUDA device is actually present and fall back
# to the NumPy FFT path everywhere if not. cupy imports fine without a GPU/driver;
# the failure surfaces only when querying devices, so we guard getDeviceCount and
# treat 0/exception as "no GPU".
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
# phase-ramp translation
# --------------------------------------------------------------------------- #
def fftTranslate2d(image, shift, arrayScale, markInvalid=True):
    """Translate `image` by `shift = (dy, dx)` via an FFT phase-ramp with pad-and-crop
    anti-wraparound, preserving the input dtype (complex64 in -> complex64 out).

    The phase ramp exp(-2j*pi*(fx*dx + fy*dy)) gives output[n] = input[n - (dy, dx)]
    -- content moves by (dy, dx) -- so fftTranslate2d(img, s, arrayScale) applies exactly the
    shift `s` (the convention getTranslationalShifts returns: applying t_i to image_i co-registers
    the set under t_j - t_i = s_ij).

    Pad by ceil(|shift|) per axis (the tight bound -- enough for any integer or
    sub-pixel shift), phase-ramp the padded FFT, inverse-FFT, then crop back to the
    original shape. Zero-pad avoids circular wraparound at the cost of mild edge
    ringing. Input NaN borders (e.g. from a prior rotation/alignment step) are handled
    explicitly: NaNs are zero-filled before the FFT so they do not propagate through
    it (an NaN anywhere in fft2 poisons the whole array), and the output's valid
    region is re-derived from the input finiteness mask shifted by (dy, dx).

    Args:
        image: (H, W) complex (or real) array, on-disk L2 convention
               (axis 0 = range/y/rows, axis 1 = azimuth/x/columns). May contain NaN
               borders (e.g. from a prior rotation step); these are zero-filled before the FFT.
        shift: (dy, dx) = (shiftY, shiftX); dy along rows (range/y), dx along cols
               (azimuth/x).
        arrayScale: required, the unit of `image` -- ``'amplitude'`` or ``'intensity'``.
            Translation is a *linear, unit-preserving* operation (a phase ramp in the
            FFT domain commutes with any per-pixel brightness mapping), so this argument
            triggers NO numeric conversion here: amplitude in -> amplitude out, intensity
            in -> intensity out, complex in -> complex out. It is required purely as the
            input-unit contract so the co-registration pipeline stays consistent end to
            end (the orchestrators convert amplitude -> intensity for the *estimation*
            branch upstream of this call; this *application* step 5 translates the
            **original** input and must preserve its declared unit).
        markInvalid: if True (default), set the output to NaN where the source has no
            real sample -- i.e. where the source index (r - dy, c - dx) is out of
            bounds OR the source pixel was NaN in the input -- so neither the fake
            zero-pad border nor the shifted input-NaN border is treated as real data
            downstream. If False, leave the shifted-out / input-NaN regions as the
            raw (zero-filled, possibly ringing) values. Note
            this marks the strictly no-source region, not the mild sinc ringing that
            bleeds a few pixels into the valid region from the zero-pad
            discontinuity (in practice tiny for band-limited SAR content).

    Returns:
        (H, W) array of the same dtype as `image`, translated by (dy, dx).
    """
    if arrayScale not in ('amplitude', 'intensity'):
        raise ValueError(f"arrayScale must be 'amplitude' or 'intensity', got {arrayScale!r}")
    image = np.asarray(image)
    H, W = image.shape
    dy = float(shift[0])
    dx = float(shift[1])

    # Input NaN handling: an NaN anywhere in fft2 propagates to the whole output, so
    # zero-fill NaNs first. Track the input finiteness mask to re-derive the output
    # valid region after the FFT (the source pixel at (r - dy, c - dx) is a real
    # sample iff it is in bounds AND was finite in the input).
    inFinite = np.isfinite(image)
    hasNaN = not inFinite.all()
    if hasNaN:
        filled = np.where(inFinite, image, 0)
    else:
        filled = image

    padY = int(np.ceil(abs(dy)))
    padX = int(np.ceil(abs(dx)))

    # crop back to the original shape from the padded array (H+2*padY, W+2*padX).
    # Explicit end indices [padY:padY+H, padX:padX+W] instead of the symmetric
    # [padY:-padY, padX:-padX]: the latter is the empty slice [0:0] when a pad is 0,
    # this gives [0:H] (a no-op).
    cropR = slice(padY, padY + H)
    cropC = slice(padX, padX + W)

    fy = np.fft.fftfreq(H + 2 * padY)[:, None]
    fx = np.fft.fftfreq(W + 2 * padX)[None, :]
    phase = np.exp(-2j * np.pi * (fx * dx + fy * dy))

    if _HAVE_CUPY_GPU:
        print("Using cupy for fft translation")
        gpuImg = cp.asarray(filled)
        padded = cp.pad(gpuImg, ((padY, padY), (padX, padX)), mode='constant')
        translated = cp.fft.ifft2(cp.fft.fft2(padded) * cp.asarray(phase))[cropR, cropC]
        translated = cp.asnumpy(translated)
        del gpuImg, padded
        cp.get_default_memory_pool().free_all_blocks()
    else:
        padded = np.pad(filled, ((padY, padY), (padX, padX)), mode='constant')
        translated = np.fft.ifft2(np.fft.fft2(padded) * phase)[cropR, cropC]

    # real input -> keep the output real (the phase-ramp IFFT has ~0 imaginary
    # residual; take magnitude).
    if not np.issubdtype(image.dtype, np.complexfloating):
        translated = np.abs(translated)

    if markInvalid:
        # output[r, c] = input[r - dy, c - dx] is a real sample iff the source index
        # is in bounds AND the source pixel was finite. The in-bounds check uses the
        # exact fractional source index; the source-finite check gathers the input
        # finiteness mask at the rounded source index (a <=1px boundary
        # approximation of a binary mask -- fine for marking no-data regions).
        srcR = np.arange(H) - dy
        srcC = np.arange(W) - dx
        oobR = (srcR < 0) | (srcR >= H)
        oobC = (srcC < 0) | (srcC >= W)
        ri = np.clip(np.rint(srcR).astype(np.intp), 0, H - 1)
        ci = np.clip(np.rint(srcC).astype(np.intp), 0, W - 1)
        valid = inFinite[np.ix_(ri, ci)]
        if oobR.any():
            valid[oobR, :] = False
        if oobC.any():
            valid[:, oobC] = False
        translated = np.where(valid, translated, np.nan)
    # markInvalid=False: leave the shifted-out / input-NaN regions as the raw
    # (zero-filled, possibly ringing) values.

    # preserve input dtype (np.fft upcasts complex64 -> complex128, float32 -> complex128)
    return translated.astype(image.dtype)