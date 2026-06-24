import numpy as np

# cupy is intentionally NOT a declared dependency (it ships as several
# mutually-exclusive wheels — cupy-cuda{11,12,13}x toolkitless, or self-contained
# cupy — that all install the `cupy` package, so the right one is host-specific;
# install via setup_cupy.py). A *usable GPU* is not required: detect at import
# whether a CUDA device is actually present and fall back to the NumPy FFT path
# if not. cupy imports fine without a GPU/CUDA driver; the failure surfaces only
# when querying devices, so we guard getDeviceCount and treat 0/exception as
# "no GPU". The module never hard-fails on a CPU-only host.
try:
    import cupy as cp
    try:
        _HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAVE_CUPY_GPU = False
except ImportError:
    cp = None
    _HAVE_CUPY_GPU = False


def _fourierUpsample_cupy(a, up, axis):
    """cupy backend for fourierUpsample (`a` is a cupy.ndarray; returns cupy).

    Same zero-padded-spectrum algorithm as the NumPy path, but the FFT/shift/pad
    run on the GPU via cuFFT. complex64 stays complex64 (cuFFT C2C single).
    """
    n = a.shape[axis]
    m = n * up
    F = cp.fft.fft(a, axis=axis)
    Fs = cp.fft.fftshift(F, axes=axis)
    pad = m - n
    padWidth = [(0, 0)] * a.ndim
    padWidth[axis] = (pad // 2, pad - pad // 2)
    Fs = cp.pad(Fs, padWidth, mode='constant')
    Fp = cp.fft.ifftshift(Fs, axes=axis)
    return cp.fft.ifft(Fp, axis=axis) * up


def fourierUpsample(a, up, axis):
    '''Upsample array `a` by an integer factor `up` along `axis` via FFT zero-padding
    (band-limited interpolation): take the FFT along `axis`, zero-pad the spectrum
    symmetrically to length `up * n`, then inverse-FFT and scale.

    This is O(N log N) and fast for the 2x per-axis upsampling used here. The
    result is an ideal band-limited interpolation (periodic-edge assumption),
    which differs slightly from a flattop-windowed sinc near edges and strong
    discontinuities.

    Dispatches on the input array type: a cupy.ndarray is upsampled on the GPU
    (cuFFT) and returned as a cupy array; anything else uses the NumPy FFT path
    and returns a NumPy array. So a caller that wants GPU acceleration moves the
    array to the GPU once with cupy.asarray and keeps it there across the
    chained per-axis upsamples — the ~1 GB intermediate never crosses PCIe.
    (GPU use is gated on a device actually being present; without one the cupy
    branch is never taken.)
    '''
    if _HAVE_CUPY_GPU and isinstance(a, cp.ndarray):
        return _fourierUpsample_cupy(a, up, axis)

    n = a.shape[axis]
    m = n * up
    F = np.fft.fft(a, axis=axis)
    Fs = np.fft.fftshift(F, axes=axis)
    pad = m - n
    padWidth = [(0, 0)] * a.ndim
    padWidth[axis] = (pad // 2, pad - pad // 2)
    Fs = np.pad(Fs, padWidth, mode='constant')
    Fp = np.fft.ifftshift(Fs, axes=axis)
    return np.fft.ifft(Fp, axis=axis) * up


def _fourierUpsample_selfcheck():
    """Equivalence check: cupy fourierUpsample vs the NumPy reference.

    cuFFT and NumPy's FFT use different backends, so the complex64 results agree
    to ~1e-5 relative but not bit-for-bit; we check allclose, not array_equal.
    Skipped (and reported as such) when no GPU is available.
    """
    if not _HAVE_CUPY_GPU:
        print("fourierUpsample self-check: skipped (no GPU available — NumPy path only).")
        return True

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, 48)) + 1j * rng.standard_normal((64, 48))).astype(np.complex64)

    # reference: both upsamples on NumPy
    ref = fourierUpsample(fourierUpsample(a, 2, axis=1), 2, axis=0)
    # gpu: move up once, upsample both axes on GPU, bring back once
    g = cp.asarray(a)
    g = fourierUpsample(g, 2, axis=1)
    g = fourierUpsample(g, 2, axis=0)
    got = cp.asnumpy(g)

    shape_ok = ref.shape == got.shape == (a.shape[0] * 2, a.shape[1] * 2)
    dtype_ok = got.dtype == ref.dtype == np.complex64
    vals_close = np.allclose(got, ref, rtol=1e-4, atol=1e-5)
    max_diff = float(np.nanmax(np.abs(got - ref)))

    ok = bool(shape_ok and dtype_ok and vals_close)
    print(f"fourierUpsample self-check: {'PASS' if ok else 'FAIL'} "
          f"(shape={shape_ok}, dtype={dtype_ok}, values_close={vals_close}, "
          f"max_abs_diff={max_diff:.3e})")
    if not ok:
        print("ref shape", ref.shape, "got shape", got.shape)
    return ok


if __name__ == "__main__":
    _fourierUpsample_selfcheck()