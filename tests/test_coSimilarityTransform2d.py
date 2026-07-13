"""Tests for ``coSimilarityTransform2d`` (NumPy/CPU path).

See ``src/imageProcToolkit/coSimilarityTransform2d.py`` and the "Testing" section
of ``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.coSimilarityTransform2d import coSimilarityTransform2d


def testCoSimilarityTransform2d_invalidArrayScaleRaises():
    imgs = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
    try:
        coSimilarityTransform2d(imgs, arrayScale="bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid arrayScale")


def testCoSimilarityTransform2d_realAmplitudeStack():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    imgs = [base, np.roll(base, shift=(1, -2), axis=(0, 1))]
    transformed, params, diag = coSimilarityTransform2d(imgs, arrayScale="amplitude")
    assert params.shape == (2, 4)
    assert len(transformed) == 2
    assert "rotScale" in diag and "translation" in diag
    assert transformed[0].dtype == np.float32


def testCoSimilarityTransform2d_complexIntensityStack():
    rng = np.random.default_rng(1)
    base = (rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))).astype(np.complex64)
    imgs = [base, np.roll(base, shift=(1, 1), axis=(0, 1))]
    transformed, params, diag = coSimilarityTransform2d(imgs, arrayScale="intensity")
    assert params.shape == (2, 4)
    assert transformed[0].dtype == np.complex64


def testCoSimilarityTransform2d_explicitMasks():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((16, 16)).astype(np.float32)
    imgs = [base, np.roll(base, shift=(1, 0), axis=(0, 1))]
    masks = [np.ones((16, 16), dtype=bool) for _ in range(2)]
    transformed, params, diag = coSimilarityTransform2d(imgs, arrayScale="amplitude", masks=masks)
    assert params.shape == (2, 4)


def testCoSimilarityTransform2d_masterIndexStarEstimation():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    imgs = [
        base,
        np.roll(base, shift=(1, -2), axis=(0, 1)),
        np.roll(base, shift=(-1, 2), axis=(0, 1)),
    ]
    _, params_m, diag_m = coSimilarityTransform2d(imgs, arrayScale="amplitude", masterIndex=-1)
    assert params_m.shape == (3, 4)
    assert np.all(params_m[-1] == 0.0)  # last image pinned at identity
    assert diag_m["rotScale"]["nPairs"] == 2  # star: n-1 pairs per stage
    assert diag_m["translation"]["nPairs"] == 2
    assert diag_m["rotScale"]["residualMax_rot_rad"] < 1e-6  # star rot/scale is exact


def testCoSimilarityTransform2d_selfcheck_passes():
    from imageProcToolkit.coSimilarityTransform2d import _coSimilarityTransform2d_selfcheck

    assert _coSimilarityTransform2d_selfcheck() is True
