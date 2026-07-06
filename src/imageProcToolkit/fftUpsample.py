import numpy as np

# cupy is intentionally NOT a declared dependency (it ships as several
# mutually-exclusive wheels — cupy-cuda{11,12,13}x toolkitless, or self-contained
# cupy — that all install the `cupy` package, so the right one is host-specific;
# install the wheel matching your host CUDA toolkit. A *usable GPU* is not
# required: detect at import whether a CUDA device is actually present and fall
# back to the NumPy FFT path if not. cupy imports fine without a GPU/CUDA driver;
# the failure surfaces only when querying devices, so we guard getDeviceCount and
# treat 0/exception as "no GPU". The module never hard-fails on a CPU-only host.
try:
    import cupy as cp
    try:
        _HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAVE_CUPY_GPU = False
except ImportError:
    cp = None
    _HAVE_CUPY_GPU = False


def _fourierUpsample_core(a, up, axis, xp):
    """``xp``-parameterized FFT zero-padding upsample core (numpy or cupy).

    Same zero-padded-spectrum algorithm on both backends: FFT along `axis`,
    fftshift, zero-pad the spectrum symmetrically to length `up * n`, ifftshift,
    inverse-FFT, scale. complex64 stays complex64 (cuFFT C2C single on the GPU).
    ``xp.asarray(a)`` is a no-op on the NumPy path and the host->device move on
    the cupy path, so the public ``fourierUpsample`` can pass whichever backend it
    dispatches to without a separate transfer step.
    """
    arr = xp.asarray(a)
    n = arr.shape[axis]
    m = n * up
    F = xp.fft.fft(arr, axis=axis)
    Fs = xp.fft.fftshift(F, axes=axis)
    pad = m - n
    padWidth = [(0, 0)] * arr.ndim
    padWidth[axis] = (pad // 2, pad - pad // 2)
    Fs = xp.pad(Fs, padWidth, mode='constant')
    Fp = xp.fft.ifftshift(Fs, axes=axis)
    return xp.fft.ifft(Fp, axis=axis) * up


def fourierUpsample(a, up, axis):
    '''Upsample array `a` by an integer factor `up` along `axis` via FFT zero-padding
    (band-limited interpolation): take the FFT along `axis`, zero-pad the spectrum
    symmetrically to length `up * n`, then inverse-FFT and scale.

    This is O(N log N) and fast for the 2x per-axis upsampling used here. The
    result is an ideal band-limited interpolation (periodic-edge assumption),
    which differs slightly from a flattop-windowed sinc near edges and strong
    discontinuities.

    Auto-accelerates on a CUDA device when present (the cross-repo convention):
    a cupy input is upsampled on the GPU (cuFFT) and returned as a cupy array
    (cupy in -> cupy out); a NumPy input is upsampled on the GPU and returned as
    a NumPy array (numpy in -> numpy out, auto round-tripped via cp.asnumpy). So
    a caller that wants to keep the array resident across chained per-axis
    upsamples passes a cupy array — the ~1 GB intermediate never crosses PCIe.
    Without a GPU the NumPy FFT path is used regardless.
    '''
    if _HAVE_CUPY_GPU:
        print("Using cupy for fourier upsample")
        out = _fourierUpsample_core(a, up, axis, cp)
        if isinstance(a, cp.ndarray):
            return out                 # cupy in -> cupy out
        return cp.asnumpy(out)         # numpy in -> numpy out
    return _fourierUpsample_core(a, up, axis, np)


def _fourierUpsample_cupy_selfcheck():
    """Equivalence check: cupy fourierUpsample vs the NumPy reference.

    The NumPy reference is produced by calling the ``xp``-parameterized
    ``_fourierUpsample_core`` with ``np`` directly (bypassing ``fourierUpsample``'s
    auto-dispatch, which uses the GPU whenever one is present). The GPU result
    comes from the public ``fourierUpsample`` with a cupy input (cupy in -> cupy
    out), brought back with ``cp.asnumpy``. The numpy-in -> numpy-out contract is
    also exercised. cuFFT and NumPy's FFT use different backends, so the
    complex64 results agree to ~1e-5 relative but not bit-for-bit; we check
    allclose, not array_equal. Skipped (and reported as such) when no GPU.
    """
    if not _HAVE_CUPY_GPU:
        print("fourierUpsample self-check: skipped (no GPU available — NumPy path only).")
        return True

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, 48)) + 1j * rng.standard_normal((64, 48))).astype(np.complex64)

    # NumPy reference: the core directly with np (bypasses auto-dispatch). The
    # core signature is (a, up, axis, xp); chain the two per-axis upsamples.
    ref = _fourierUpsample_core(_fourierUpsample_core(a, 2, 1, np), 2, 0, np)
    # GPU via the public function: cupy in -> cupy out across both axes, then
    # bring back once.
    g = cp.asarray(a)
    g = fourierUpsample(g, 2, axis=1)
    g = fourierUpsample(g, 2, axis=0)
    got = cp.asnumpy(g)
    # numpy in -> numpy out contract: a NumPy input must return a NumPy array.
    nio = fourierUpsample(a, 2, axis=1)
    nio_ok = isinstance(nio, np.ndarray)

    shape_ok = ref.shape == got.shape == (a.shape[0] * 2, a.shape[1] * 2)
    dtype_ok = got.dtype == ref.dtype == np.complex64
    vals_close = np.allclose(got, ref, rtol=1e-4, atol=1e-5)
    max_diff = float(np.nanmax(np.abs(got - ref)))

    ok = bool(shape_ok and dtype_ok and vals_close and nio_ok)
    print(f"fourierUpsample self-check: {'PASS' if ok else 'FAIL'} "
          f"(shape={shape_ok}, dtype={dtype_ok}, values_close={vals_close}, "
          f"nio={nio_ok}, max_abs_diff={max_diff:.3e})")
    if not ok:
        print("ref shape", ref.shape, "got shape", got.shape)
    return ok


if __name__ == "__main__":
    _fourierUpsample_cupy_selfcheck()