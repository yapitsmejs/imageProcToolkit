"""Tests for ``coTranslate2d`` (NumPy/CPU path).

See ``src/imageProcToolkit/coTranslate2d.py`` and the "Testing" section of
``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.coTranslate2d import coTranslate2d, fftTranslateImages


def testCoTranslate2d_invalidArrayScaleRaises():
    imgs = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
    try:
        coTranslate2d(imgs, arrayScale="wrong")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid arrayScale")


def testCoTranslate2d_realAmplitudeStack():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    imgs = [base, np.roll(base, shift=(2, -3), axis=(0, 1))]
    translated, shifts, diag = coTranslate2d(imgs, arrayScale="amplitude")
    assert shifts.shape == (2, 2)
    assert len(translated) == 2
    assert "nPairs" in diag
    # default zero-mean gauge
    assert abs(shifts.sum()) < 1e-6
    # complex/real preserved: real in -> real out
    assert translated[0].dtype == np.float32


def testCoTranslate2d_complexIntensityStack():
    rng = np.random.default_rng(1)
    base = (rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))).astype(np.complex64)
    imgs = [base, np.roll(base, shift=(1, 1), axis=(0, 1))]
    translated, shifts, diag = coTranslate2d(imgs, arrayScale="intensity")
    assert shifts.shape == (2, 2)
    assert translated[0].dtype == np.complex64


def testCoTranslate2d_masterIndexStarEstimation():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    imgs = [
        base,
        np.roll(base, shift=(2, -3), axis=(0, 1)),
        np.roll(base, shift=(-1, 4), axis=(0, 1)),
    ]
    _, shifts_zm, diag_zm = coTranslate2d(imgs, arrayScale="amplitude")
    assert abs(shifts_zm.sum()) < 1e-6  # default: zero-mean gauge
    assert diag_zm["nPairs"] == 3  # default: all pairs (n=3 -> 3)

    _, shifts_m, diag_m = coTranslate2d(imgs, arrayScale="amplitude", masterIndex=-1)
    assert shifts_m.shape == (3, 2)
    assert np.all(shifts_m[-1] == 0.0)  # last image pinned at identity
    assert diag_m["nPairs"] == 2  # star: n-1 pairs, not 3
    assert diag_m["residualMax_px"] < 1e-6  # star + fix-node is exact


def testCoTranslate2d_explicitMasks():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((16, 16)).astype(np.float32)
    imgs = [base, np.roll(base, shift=(1, 1), axis=(0, 1))]
    masks = [np.ones((16, 16), dtype=bool) for _ in range(2)]
    translated, shifts, diag = coTranslate2d(imgs, arrayScale="amplitude", masks=masks)
    assert shifts.shape == (2, 2)


def testFftTranslateImages_invalidArrayScaleRaises():
    imgs = [np.zeros((4, 4), dtype=np.float32)]
    shifts = np.array([[0.0, 0.0]])
    try:
        fftTranslateImages(imgs, shifts, arrayScale="bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid arrayScale")


def testFftTranslateImages_countMismatchRaises():
    imgs = [np.zeros((4, 4), dtype=np.float32) for _ in range(2)]
    shifts = np.array([[0.0, 0.0]])
    try:
        fftTranslateImages(imgs, shifts, arrayScale="amplitude")
    except ValueError:
        return
    raise AssertionError("expected ValueError for image/shift count mismatch")
