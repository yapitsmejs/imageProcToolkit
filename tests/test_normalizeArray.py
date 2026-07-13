"""Tests for ``normalizeArray`` (NumPy/CPU path).

See ``src/imageProcToolkit/normalizeArray.py`` and the "Testing" section of
``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.normalizeArray import normalizeToUint8


def testNormalizeToUint8_realInput():
    img = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    out = normalizeToUint8(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    # min -> 0, max -> 255
    assert out[0, 0] == 0
    assert out[1, 2] == 255


def testNormalizeToUint8_complexInputUsesAmplitude():
    img = np.array([[0.0 + 0.0j, 3.0 + 4.0j]], dtype=np.complex64)
    out = normalizeToUint8(img)
    assert out.dtype == np.uint8
    # |3+4j| = 5 is the max; 0 is the min -> [0, 255]
    assert out[0, 0] == 0
    assert out[0, 1] == 255


def testNormalizeToUint8_allNaNReturnsZeros():
    img = np.full((3, 3), np.nan, dtype=np.float32)
    out = normalizeToUint8(img)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def testNormalizeToUint8_degenerateRangeReturnsZeros():
    img = np.full((3, 3), 2.5, dtype=np.float32)
    out = normalizeToUint8(img)
    assert np.all(out == 0)


def testNormalizeToUint8_nanMapsToZero():
    img = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32)
    out = normalizeToUint8(img)
    assert out[0, 1] == 0  # NaN -> sentinel 0
    assert out[1, 1] == 255  # max -> 255


def testNormalizeToUint8_integerInput():
    img = np.array([[0, 5, 10]], dtype=np.int16)
    out = normalizeToUint8(img)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0
    assert out[0, 2] == 255
