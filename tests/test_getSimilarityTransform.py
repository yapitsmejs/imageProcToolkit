"""Tests for ``getSimilarityTransform`` (NumPy/CPU path).

See ``src/imageProcToolkit/getSimilarityTransform.py`` and the "Testing" section
of ``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.getSimilarityTransform import (
    allPairwiseSimilarityTransforms,
    checkSimilarityRotScaleResiduals,
    getSimilarityTransform,
    pairwiseSimilarityTransform,
    solveGlobalSimilarityRotScale,
)


def testPairwiseSimilarityTransform_returnsFourTuple():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = np.roll(A, shift=(1, 0), axis=(0, 1))
    theta, log_s, peakHeight, conf = pairwiseSimilarityTransform(A, B, subpixel=False)
    assert np.isfinite(theta)
    assert np.isfinite(log_s)
    assert np.isfinite(peakHeight)
    assert np.isfinite(conf)


def testPairwiseSimilarityTransform_highPassFalseBranch():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = np.roll(A, shift=(1, 0), axis=(0, 1))
    theta, log_s, _ph, _conf = pairwiseSimilarityTransform(A, B, subpixel=False, highPass=False)
    assert np.isfinite(theta)
    assert np.isfinite(log_s)


def testPairwiseSimilarityTransform_explicitNrhoNphi():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = np.roll(A, shift=(1, 0), axis=(0, 1))
    theta, log_s, _ph, _conf = pairwiseSimilarityTransform(A, B, subpixel=False, nRho=32, nPhi=180)
    assert np.isfinite(theta)
    assert np.isfinite(log_s)


def testPairwiseSimilarityTransform_emptyMaskUsesZeroMean():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = np.roll(A, shift=(1, 0), axis=(0, 1))
    emptyMask = np.zeros((16, 16), dtype=bool)
    theta, log_s, _ph, _conf = pairwiseSimilarityTransform(
        A, B, maskA=emptyMask, maskB=emptyMask, subpixel=False
    )
    assert np.isfinite(theta)


def testPairwiseSimilarityTransform_tinyImageRmaxFallback():
    # 2x2 image: rmax <= rmin triggers the rmax = rmin + 1.0 fallback
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.roll(A, shift=(1, 0), axis=(0, 1))
    theta, log_s, _ph, _conf = pairwiseSimilarityTransform(A, B, subpixel=False)
    assert np.isfinite(theta)


def testPairwiseSimilarityTransform_subpixelBranch():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = np.roll(A, shift=(1, 0), axis=(0, 1))
    theta, log_s, _ph, _conf = pairwiseSimilarityTransform(A, B, subpixel=True)
    assert np.isfinite(theta)
    assert np.isfinite(log_s)


def testAllPairwiseSimilarityTransforms_defaultAllPairs():
    imgs = [np.zeros((16, 16), dtype=np.uint8) + 1 for _ in range(3)]
    pw = allPairwiseSimilarityTransforms(imgs, subpixel=False)
    assert set(pw.keys()) == {(0, 1), (0, 2), (1, 2)}


def testAllPairwiseSimilarityTransforms_pairsFilter():
    imgs = [np.zeros((16, 16), dtype=np.uint8) + 1 for _ in range(3)]
    pw = allPairwiseSimilarityTransforms(imgs, pairs=[(0, 1)], subpixel=False)
    assert set(pw.keys()) == {(0, 1)}


def testSolveGlobalSimilarityRotScale_zeroMean():
    pw = {(0, 1): (0.1, 0.2, 1.0, 1.0)}
    p = solveGlobalSimilarityRotScale(pw, n=2)
    assert p.shape == (2, 2)
    assert np.isclose(p.sum(), 0.0)


def testSolveGlobalSimilarityRotScale_masterIndex():
    pw = {
        (0, 1): (0.1, 0.2, 1.0, 1.0),
        (0, 2): (0.0, 0.0, 1.0, 1.0),
        (1, 2): (-0.1, -0.2, 1.0, 1.0),
    }
    p = solveGlobalSimilarityRotScale(pw, n=3, masterIndex=0)
    assert np.allclose(p[0], [0.0, 0.0])


def testCheckSimilarityRotScaleResiduals_empty():
    diag = checkSimilarityRotScaleResiduals({}, np.zeros((2, 2)))
    assert diag["nPairs"] == 0
    assert diag["residualMax_rot_rad"] == 0.0


def testCheckSimilarityRotScaleResiduals_residuals():
    pw = {(0, 1): (0.1, 0.2, 1.0, 1.0)}
    params = np.array([[-0.05, -0.1], [0.05, 0.1]])
    diag = checkSimilarityRotScaleResiduals(pw, params)
    assert diag["nPairs"] == 1
    assert np.isfinite(diag["residualMax_rot_rad"])


def testGetSimilarityTransform_shape(capsys):
    rng = np.random.default_rng(0)
    imgs = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
    params = getSimilarityTransform(imgs, subpixel=False)
    assert params.shape == (2, 4)
    captured = capsys.readouterr()
    assert "getSimilarityTransform" in captured.out


def testGetSimilarityTransform_selfcheck_passes():
    from imageProcToolkit.getSimilarityTransform import _getSimilarityTransform_selfcheck

    assert _getSimilarityTransform_selfcheck() is True
