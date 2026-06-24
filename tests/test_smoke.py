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
from imageProcToolkit.getSimilarityTransform import getSimilarityTransform
from imageProcToolkit.similarityTransformImage import (
    similarityTransformImage, similarityTransformImages)
from imageProcToolkit.coSimilarityTransformImages import coSimilarityTransformImages


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
    import imageProcToolkit._phaseCorrelationCore  # noqa: F401
    import imageProcToolkit.getSimilarityTransform  # noqa: F401
    import imageProcToolkit.similarityTransformImage  # noqa: F401
    import imageProcToolkit.coSimilarityTransformImages  # noqa: F401


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


def test_getSimilarityTransform_shape():
    # Existence + shape sanity; relative imports inside getSimilarityTransform
    # (-> _phaseCorrelationCore/interp2/getTranslationalShifts/clamp/norm/
    # similarityTransformImage) are what we really want to confirm here.
    assert callable(getSimilarityTransform)
    rng = np.random.default_rng(0)
    imgs = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
    params = getSimilarityTransform(imgs, subpixel=False)
    assert params.shape == (2, 4)


def test_similarityTransformImage_shape_dtype():
    # atomic warp: shape preserved, dtype preserved for real and complex input.
    img = np.zeros((16, 16), dtype=np.float32)
    img[8, 8] = 1.0
    out = similarityTransformImage(img, (0.1, 1.05, 1.0, -0.5))
    assert out.shape == img.shape
    assert out.dtype == np.float32
    cplx = (np.zeros((16, 16)) + 1j * np.zeros((16, 16))).astype(np.complex64)
    cplx_out = similarityTransformImage(cplx, (0.05, 1.02, 0.5, 0.5))
    assert cplx_out.dtype == np.complex64


def test_coSimilarityTransformImages_callable():
    # orchestrator: returns (transformed, params(N,4), diag with rotScale + translation).
    assert callable(coSimilarityTransformImages)
    rng = np.random.default_rng(0)
    imgs = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
    transformed, params, diag = coSimilarityTransformImages(imgs)
    assert params.shape == (2, 4)
    assert len(transformed) == 2
    assert 'rotScale' in diag and 'translation' in diag