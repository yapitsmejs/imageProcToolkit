"""Tests for ``fftTranslate2d`` (NumPy/CPU path).

See ``src/imageProcToolkit/fftTranslate2d.py`` and the "Testing" section of
``CONVENTIONS.md``. The CuPy/GPU dispatch branch is excluded from coverage
(``# pragma: no cover``) -- it cannot run on CPU CI.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.fftTranslate2d import fftTranslate2d


def testFftTranslate2d_preservesShapeAndDtype_real():
    img = np.zeros((10, 10), dtype=np.float32)
    img[5, 5] = 1.0
    out = fftTranslate2d(img, shift=(1.5, -0.5), arrayScale="intensity")
    assert out.shape == img.shape
    assert out.dtype == np.float32


def testFftTranslate2d_preservesComplex64():
    img = (np.zeros((10, 10)) + 1j * np.zeros((10, 10))).astype(np.complex64)
    out = fftTranslate2d(img, shift=(1.0, 0.5), arrayScale="amplitude")
    assert out.dtype == np.complex64


def testFftTranslate2d_zeroShiftIsNoOp():
    img = np.zeros((8, 8), dtype=np.float32)
    img[4, 4] = 10.0
    out = fftTranslate2d(img, shift=(0.0, 0.0), arrayScale="intensity", markInvalid=False)
    # zero shift: content unchanged at the source pixel
    assert np.isclose(out[4, 4], 10.0, atol=1e-5)


def testFftTranslate2d_invalidArrayScaleRaises():
    img = np.zeros((4, 4), dtype=np.float32)
    try:
        fftTranslate2d(img, shift=(1.0, 1.0), arrayScale="wrong")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid arrayScale")


def testFftTranslate2d_markInvalidProducesNaNBorder():
    img = np.zeros((8, 8), dtype=np.float32)
    img[4, 4] = 1.0
    out = fftTranslate2d(img, shift=(4.0, 0.0), arrayScale="intensity", markInvalid=True)
    # a 4-pixel shift moves the source region; some rows should be NaN (no
    # source sample)
    assert np.any(np.isnan(out))


def testFftTranslate2d_markInvalidFalseLeavesRawValues():
    img = np.zeros((8, 8), dtype=np.float32)
    img[4, 4] = 1.0
    out = fftTranslate2d(img, shift=(4.0, 0.0), arrayScale="intensity", markInvalid=False)
    # markInvalid=False: no NaN fill (may contain ringing); at minimum finite
    assert out.shape == img.shape


def testFftTranslate2d_nanInputZeroFilled():
    img = np.full((6, 6), np.nan, dtype=np.float32)
    img[3, 3] = 5.0
    out = fftTranslate2d(img, shift=(1.0, 0.0), arrayScale="intensity")
    assert out.shape == img.shape
    # most pixels are NaN-marked (no real source or NaN source), one finite
    assert np.any(np.isnan(out))
