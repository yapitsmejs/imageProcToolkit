"""Tests for ``getTranslationalShifts`` (NumPy/CPU path).

See ``src/imageProcToolkit/getTranslationalShifts.py`` and the "Testing" section
of ``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.getTranslationalShifts import (
    _solveLaplacianGauge,
    allPairwiseTranslationalShifts,
    checkTranslationalShiftResiduals,
    getTranslationalShifts,
    pairwiseTranslationalShift,
    solveGlobalTranslationalShifts,
)


def _identicalShiftImages(shift: tuple[int, int], size: int = 16) -> list[np.ndarray]:
    base = np.zeros((size, size), dtype=np.uint8)
    base[size // 2, size // 2] = 200
    return [base, np.roll(base, shift=shift, axis=(0, 1))]


def testPairwiseTranslationalShift_returnsShiftAppliedToB():
    A = np.zeros((16, 16), dtype=np.uint8)
    A[8, 8] = 200
    B = np.roll(A, shift=(2, -3), axis=(0, 1))
    dy, dx, _ph, _conf = pairwiseTranslationalShift(A, B, subpixel=False)
    # B was rolled by (2, -3); the shift to apply to B to align to A is (-2, 3)
    assert abs(dy + 2) < 1.0
    assert abs(dx - 3) < 1.0


def testPairwiseTranslationalShift_subpixelBranch():
    A = np.zeros((16, 16), dtype=np.uint8)
    A[8, 8] = 200
    B = np.roll(A, shift=(2, -3), axis=(0, 1))
    dy, dx, _ph, _conf = pairwiseTranslationalShift(A, B, subpixel=True)
    assert abs(dy + 2) < 1.0
    assert abs(dx - 3) < 1.0


def testAllPairwiseTranslationalShifts_defaultAllPairs():
    imgs = [np.zeros((8, 8), dtype=np.uint8) for _ in range(3)]
    imgs[0][4, 4] = 200
    imgs[1] = np.roll(imgs[0], shift=(1, 0), axis=(0, 1))
    imgs[2] = np.roll(imgs[0], shift=(0, 1), axis=(0, 1))
    pw = allPairwiseTranslationalShifts(imgs, subpixel=False)
    assert set(pw.keys()) == {(0, 1), (0, 2), (1, 2)}


def testAllPairwiseTranslationalShifts_pairsFilter():
    imgs = [np.zeros((8, 8), dtype=np.uint8) for _ in range(3)]
    pw = allPairwiseTranslationalShifts(imgs, pairs=[(0, 1)], subpixel=False)
    assert set(pw.keys()) == {(0, 1)}


def testSolveLaplacianGauge_zeroMeanGivesZeroSum():
    # one edge s_01 = 5 -> zero-mean gauge gives t = (-2.5, 2.5)
    L = np.array([[1.0, -1.0], [-1.0, 1.0]])
    d = np.array([[5.0], [-5.0]])
    p = _solveLaplacianGauge(L, d, masterIndex=None)
    assert np.isclose(p.sum(), 0.0)
    assert np.allclose(p.ravel(), [-2.5, 2.5])


def testSolveLaplacianGauge_masterIndexPinsNode():
    L = np.array([[1.0, -1.0], [-1.0, 1.0]])
    d = np.array([[5.0], [-5.0]])
    p = _solveLaplacianGauge(L, d, masterIndex=0)
    assert np.allclose(p[0], [0.0])
    assert np.allclose(p[1], [5.0])


def testSolveGlobalTranslationalShifts_zeroMean():
    pw = {(0, 1): (5.0, 0.0, 1.0, 1.0)}
    t = solveGlobalTranslationalShifts(pw, n=2)
    assert np.isclose(t.sum(), 0.0)


def testSolveGlobalTranslationalShifts_masterIndex():
    pw = {(0, 1): (5.0, 0.0, 1.0, 1.0), (0, 2): (3.0, 1.0, 1.0, 1.0), (1, 2): (-2.0, 1.0, 1.0, 1.0)}
    t = solveGlobalTranslationalShifts(pw, n=3, masterIndex=0)
    assert np.allclose(t[0], [0.0, 0.0])


def testSolveGlobalTranslationalShifts_negativeMasterWraps():
    pw = {(0, 1): (5.0, 0.0, 1.0, 1.0)}
    t = solveGlobalTranslationalShifts(pw, n=2, masterIndex=-1)
    assert np.allclose(t[1], [0.0, 0.0])


def testCheckTranslationalShiftResiduals_emptyPairwise():
    diag = checkTranslationalShiftResiduals({}, np.zeros((2, 2)))
    assert diag["nPairs"] == 0
    assert diag["residualMax_px"] == 0.0
    assert diag["residualMean_px"] == 0.0


def testCheckTranslationalShiftResiduals_residuals():
    pw = {(0, 1): (5.0, 0.0, 1.0, 1.0)}
    shifts = np.array([[-2.5, 0.0], [2.5, 0.0]])
    diag = checkTranslationalShiftResiduals(pw, shifts)
    assert diag["nPairs"] == 1
    # res_ij = s_ij - (t_j - t_i) = (5,0) - (5,0) = 0
    assert np.isclose(diag["residualMax_px"], 0.0)


def testGetTranslationalShifts_printsDiagnostics(capsys):
    imgs = _identicalShiftImages((2, 0))
    t = getTranslationalShifts(imgs, subpixel=False)
    assert t.shape == (2, 2)
    captured = capsys.readouterr()
    assert "getTranslationalShifts" in captured.out
