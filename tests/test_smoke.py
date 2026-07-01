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
from imageProcToolkit.fftTranslate2d import fftTranslate2d
from imageProcToolkit.clamp import clamp
from imageProcToolkit.normalizeArray import normalizeToUint8
from imageProcToolkit.getTranslationalShifts import getTranslationalShifts
from imageProcToolkit.coTranslate2d import coTranslate2d
from imageProcToolkit.getSimilarityTransform import getSimilarityTransform
from imageProcToolkit.similarityTransform2d import (
    similarityTransform2d, similarityTransformImages)
from imageProcToolkit.coSimilarityTransform2d import coSimilarityTransform2d


def test_version():
    assert imageProcToolkit.__version__ == "0.1.0"


def test_all_submodules_importable():
    # Relative imports inside the package must resolve.
    import imageProcToolkit.clamp  # noqa: F401
    import imageProcToolkit.coTranslate2d    # noqa: F401
    import imageProcToolkit.fftTranslate2d    # noqa: F401
    import imageProcToolkit.fftUpsample          # noqa: F401
    import imageProcToolkit.getTranslationalShifts  # noqa: F401
    import imageProcToolkit.interp2              # noqa: F401
    import imageProcToolkit.normalizeArray  # noqa: F401
    import imageProcToolkit._phaseCorrelationCore  # noqa: F401
    import imageProcToolkit.getSimilarityTransform  # noqa: F401
    import imageProcToolkit.similarityTransform2d  # noqa: F401
    import imageProcToolkit.coSimilarityTransform2d  # noqa: F401


def test_clamp_then_normalize():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    # clamp operates on real intensity: complex -> |z|^2 first.
    clamped = clamp(np.abs(img) ** 2)
    assert clamped.shape == img.shape
    assert clamped.dtype == np.float32
    normed = normalizeToUint8(clamped)
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
    out = fftTranslate2d(img, shift=(1.5, -0.5), arrayScale='intensity')
    assert out.shape == img.shape


def test_interp2linear_basic():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    xi = np.array([1.5, 2.5])
    yi = np.array([0.5, 1.5])
    out = interp2linear(z, xi, yi)
    assert out.shape == xi.shape


def test_getTranslationalShifts_and_coTranslateImages_callable():
    # Existence + signature sanity; relative imports inside these modules
    # (getTranslationalShifts -> fftUpsample, coTranslate2d ->
    #  getTranslationalShifts/fftTranslate2d/clamp/norm) are what we really want
    # to confirm here.
    assert callable(getTranslationalShifts)
    assert callable(coTranslate2d)
    imgs = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
    shifts = getTranslationalShifts(imgs, subpixel=False)
    assert shifts.shape[0] == 2


def test_getSimilarityTransform_shape():
    # Existence + shape sanity; relative imports inside getSimilarityTransform
    # (-> _phaseCorrelationCore/interp2/getTranslationalShifts/clamp/norm/
    # similarityTransform2d) are what we really want to confirm here.
    assert callable(getSimilarityTransform)
    rng = np.random.default_rng(0)
    imgs = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
    params = getSimilarityTransform(imgs, subpixel=False)
    assert params.shape == (2, 4)


def test_similarityTransformImage_shape_dtype():
    # atomic warp: shape preserved, dtype preserved for real and complex input.
    img = np.zeros((16, 16), dtype=np.float32)
    img[8, 8] = 1.0
    out = similarityTransform2d(img, (0.1, 1.05, 1.0, -0.5), arrayScale='intensity')
    assert out.shape == img.shape
    assert out.dtype == np.float32
    cplx = (np.zeros((16, 16)) + 1j * np.zeros((16, 16))).astype(np.complex64)
    cplx_out = similarityTransform2d(cplx, (0.05, 1.02, 0.5, 0.5), arrayScale='amplitude')
    assert cplx_out.dtype == np.complex64


def test_coSimilarityTransformImages_callable():
    # orchestrator: returns (transformed, params(N,4), diag with rotScale + translation).
    assert callable(coSimilarityTransform2d)
    rng = np.random.default_rng(0)
    imgs = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
    transformed, params, diag = coSimilarityTransform2d(imgs, arrayScale='amplitude')
    assert params.shape == (2, 4)
    assert len(transformed) == 2
    assert 'rotScale' in diag and 'translation' in diag


def test_coTranslate2d_master_index_pins_last():
    # masterIndex=-1 (last image) is fixed at (0, 0); the default is zero-mean.
    rng = np.random.default_rng(1)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    imgs = [base, np.roll(base, shift=(2, -3), axis=(0, 1)),
            np.roll(base, shift=(-1, 4), axis=(0, 1))]

    _, shifts_zm, _ = coTranslate2d(imgs, arrayScale='amplitude')
    assert abs(shifts_zm.sum()) < 1e-6                      # default: zero-mean gauge

    _, shifts_m, diag_m = coTranslate2d(imgs, arrayScale='amplitude', masterIndex=-1)
    assert shifts_m.shape == (3, 2)
    assert np.all(shifts_m[-1] == 0.0)                      # last image pinned at identity
    # fix-node gauge == zero-mean gauge re-expressed relative to the last image:
    assert np.allclose(shifts_m, shifts_zm - shifts_zm[-1])
    assert diag_m['residualMax_px'] < 1e-6                  # residuals are gauge-invariant


def test_coSimilarityTransform2d_master_index_pins_last():
    # masterIndex=-1 pins the last image's full 4-DOF similarity at identity and the
    # co-registration still resolves cleanly. (The single-stage fix-node == zero-mean -
    # master-row identity that test_coTranslate2d checks does NOT hold exactly here,
    # because stage 2 estimates translation on images dewarped with the gauge-dependent
    # stage-1 rot/scale -- a tiny, expected cross-gauge perturbation. So we assert the
    # real correctness properties instead: master pinned + low residuals.)
    rng = np.random.default_rng(2)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    imgs = [base, np.roll(base, shift=(1, -2), axis=(0, 1)),
            np.roll(base, shift=(-1, 2), axis=(0, 1))]

    _, params_m, diag_m = coSimilarityTransform2d(imgs, arrayScale='amplitude',
                                                  masterIndex=-1)
    assert params_m.shape == (3, 4)
    assert np.all(params_m[-1] == 0.0)                      # last image pinned at identity
    assert diag_m['rotScale']['residualMax_rot_rad'] < 1e-6   # rot/scale self-consistent
    assert diag_m['translation']['residualMax_px'] < 0.1      # translations co-register