"""Smoke tests: import the package and exercise the single-input callables.

These do NOT exercise the GPU (cupy) or multi-image co-registration paths — they
only confirm the src-layout package installs, imports, and that the relative
imports between submodules survived the restructuring. numba/cupy are optional at
runtime; the callables used here have NumPy fallbacks.
"""
import numpy as np

import imageProcToolkit
from imageProcToolkit.fftUpsample import fourierUpsample
from imageProcToolkit.interp2 import interp2linear
from imageProcToolkit.fftTranslateImage import fftTranslateImage
from imageProcToolkit.clampImageAmplitude import clampImageAmplitude
from imageProcToolkit.normalizeImageAmplitude import normalizeToUint8
from imageProcToolkit.getTranslationalShifts import getTranslationalShifts
from imageProcToolkit.coTranslateImages import coTranslateImages


def test_version():
    assert imageProcToolkit.__version__ == "0.1.0"


def test_all_submodules_importable():
    # Relative imports inside the package must resolve.
    import imageProcToolkit.clampImageAmplitude  # noqa: F401
    import imageProcToolkit.coTranslateImages    # noqa: F401
    import imageProcToolkit.fftTranslateImage    # noqa: F401
    import imageProcToolkit.fftUpsample          # noqa: F401
    import imageProcToolkit.getTranslationalShifts  # noqa: F401
    import imageProcToolkit.interp2              # noqa: F401
    import imageProcToolkit.normalizeImageAmplitude  # noqa: F401


def test_clamp_then_normalize():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    clamped = clampImageAmplitude(img)
    assert clamped.shape == img.shape
    assert clamped.dtype == np.float32
    normed, vmin, vmax = normalizeToUint8(clamped)
    assert normed.dtype == np.uint8
    assert normed.shape == img.shape


def test_fourierUpsample_shape():
    a = np.zeros((8, 8), dtype=np.float32)
    a[4, 4] = 1.0
    out = fourierUpsample(a, up=2, axis=0)
    assert out.shape == (16, 8)


def test_fftTranslateImage_shape():
    img = np.zeros((10, 10), dtype=np.float32)
    img[5, 5] = 1.0
    out = fftTranslateImage(img, shift=(1.5, -0.5))
    assert out.shape == img.shape


def test_interp2linear_basic():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    xi = np.array([1.5, 2.5])
    yi = np.array([0.5, 1.5])
    out = interp2linear(z, xi, yi)
    assert out.shape == xi.shape


def test_getTranslationalShifts_and_coTranslateImages_callable():
    # Existence + signature sanity; relative imports inside these modules
    # (getTranslationalShifts -> fftUpsample, coTranslateImages ->
    #  getTranslationalShifts/fftTranslateImage/clamp/norm) are what we really want
    # to confirm here.
    assert callable(getTranslationalShifts)
    assert callable(coTranslateImages)
    imgs = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
    shifts = getTranslationalShifts(imgs, subpixel=False)
    assert shifts.shape[0] == 2