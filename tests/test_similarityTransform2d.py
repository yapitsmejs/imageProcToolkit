"""Tests for ``similarityTransform2d`` (NumPy/CPU path).

See ``src/imageProcToolkit/similarityTransform2d.py`` and the "Testing" section
of ``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.similarityTransform2d import (
    similarityRotScaleImage,
    similarityTransform2d,
    similarityTransformImages,
)


def testSimilarityRotScaleImage_invalidScaleRaises():
    img = np.zeros((8, 8), dtype=np.float32)
    try:
        similarityRotScaleImage(img, (0.0, -1.0))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive scale")


def testSimilarityRotScaleImage_realFloat32():
    img = np.zeros((16, 16), dtype=np.float32)
    img[8, 8] = 1.0
    out = similarityRotScaleImage(img, (0.1, 1.05), markInvalid=True)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    # out-of-source border is NaN
    assert np.any(np.isnan(out))


def testSimilarityRotScaleImage_markInvalidFalseZeroFilled():
    img = np.zeros((16, 16), dtype=np.float32)
    img[8, 8] = 1.0
    out = similarityRotScaleImage(img, (0.2, 1.1), markInvalid=False)
    assert out.shape == img.shape
    assert out.dtype == np.float32


def testSimilarityRotScaleImage_integerInputPromoted():
    img = np.zeros((16, 16), dtype=np.uint8)
    img[8, 8] = 100
    out = similarityRotScaleImage(img, (0.1, 1.0), markInvalid=True)
    assert out.dtype == np.float32  # integer promoted to float32


def testSimilarityTransform2d_invalidArrayScaleRaises():
    img = np.zeros((8, 8), dtype=np.float32)
    try:
        similarityTransform2d(img, (0.1, 1.05, 1.0, -0.5), arrayScale="bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid arrayScale")


def testSimilarityTransform2d_realShapeDtypePreserved():
    img = np.zeros((16, 16), dtype=np.float32)
    img[8, 8] = 1.0
    out = similarityTransform2d(img, (0.1, 1.05, 1.0, -0.5), arrayScale="intensity")
    assert out.shape == img.shape
    assert out.dtype == np.float32


def testSimilarityTransform2d_complex64Preserved():
    cplx = (np.zeros((16, 16)) + 1j * np.zeros((16, 16))).astype(np.complex64)
    out = similarityTransform2d(cplx, (0.05, 1.02, 0.5, 0.5), arrayScale="amplitude")
    assert out.dtype == np.complex64


def testSimilarityTransformImages_invalidArrayScaleRaises():
    imgs = [np.zeros((4, 4), dtype=np.float32)]
    params = np.array([[0.0, 0.0, 0.0, 0.0]])
    try:
        similarityTransformImages(imgs, params, arrayScale="bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid arrayScale")


def testSimilarityTransformImages_countMismatchRaises():
    imgs = [np.zeros((4, 4), dtype=np.float32) for _ in range(2)]
    params = np.array([[0.0, 0.0, 0.0, 0.0]])
    try:
        similarityTransformImages(imgs, params, arrayScale="amplitude")
    except ValueError:
        return
    raise AssertionError("expected ValueError for image/param count mismatch")


def testSimilarityTransformImages_batchApplies():
    imgs = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
    imgs[0][4, 4] = 1.0
    imgs[1][5, 5] = 1.0
    params = np.array([[0.05, 0.0, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0]])
    out = similarityTransformImages(imgs, params, arrayScale="amplitude")
    assert len(out) == 2
    assert all(o.shape == (8, 8) for o in out)


def testSimilarityTransform2d_selfcheck_passes():
    from imageProcToolkit.similarityTransform2d import _similarityTransform2d_selfcheck

    assert _similarityTransform2d_selfcheck() is True
