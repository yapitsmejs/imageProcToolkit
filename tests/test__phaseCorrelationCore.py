"""Tests for ``_phaseCorrelationCore`` (NumPy/CPU path).

See ``src/imageProcToolkit/_phaseCorrelationCore.py`` and the "Testing" section
of ``CONVENTIONS.md``. The CuPy/GPU dispatch branch is excluded from coverage
(``# pragma: no cover``) -- it cannot run on CPU CI.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit._phaseCorrelationCore import (
    _peakAndSubpixel,
    _phaseCorrelation_core,
    _phaseCorrelationMap,
    _quadFit3x3,
    _wraparoundPick,
)


def testPhaseCorrelationCore_numpyPathReturnsArray():
    A = np.zeros((8, 8), dtype=np.float32)
    B = np.zeros((8, 8), dtype=np.float32)
    A[2, 3] = 1.0
    B[2, 3] = 1.0
    out = _phaseCorrelation_core(A, B, (8, 8), np)
    assert out.shape == (8, 8)


def testPhaseCorrelationMap_alignedImagesPeakAtZero():
    A = np.zeros((16, 16), dtype=np.float32)
    A[5, 7] = 10.0
    B = A.copy()
    corrMap, nValid = _phaseCorrelationMap(A, B)
    assert corrMap.shape == (16, 16)
    assert nValid == 16 * 16
    peak = np.unravel_index(int(np.argmax(corrMap)), corrMap.shape)
    assert peak == (0, 0)


def testPhaseCorrelationMap_shiftedImagesPeakAtShift():
    A = np.zeros((16, 16), dtype=np.float32)
    A[5, 7] = 10.0
    B = np.roll(A, shift=(2, -3), axis=(0, 1))
    corrMap, _ = _phaseCorrelationMap(A, B)
    peak = np.unravel_index(int(np.argmax(corrMap)), corrMap.shape)
    dy, dx = _wraparoundPick(np.array(peak, dtype=np.float64), corrMap.shape)
    assert abs(dy - (-2)) < 1.0  # B was rolled by (2, -3) -> shift applied to B is (-2, 3)
    assert abs(dx - 3) < 1.0


def testPhaseCorrelationMap_emptyMasksReturnsValidCountZero():
    A = np.zeros((8, 8), dtype=np.float32)
    B = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    _, nValid = _phaseCorrelationMap(A, B, maskA=mask, maskB=mask)
    assert nValid == 0


def testPhaseCorrelationMap_cupySelfcheck_runsOnCpuAndReportsSkip(capsys):
    from imageProcToolkit._phaseCorrelationCore import _phaseCorrelationMap_cupy_selfcheck

    assert _phaseCorrelationMap_cupy_selfcheck() is True
    captured = capsys.readouterr()
    assert "skipped" in captured.out


def testQuadFit3x3_recoversCenter():
    # a paraboloid peaked at (0, 0)
    px, py = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    patch = -(px**2 + py**2)
    dyOff, dxOff = _quadFit3x3(patch)
    assert np.isclose(dyOff, 0.0, atol=1e-6)
    assert np.isclose(dxOff, 0.0, atol=1e-6)


def testQuadFit3x3_degenerateReturnsZero():
    patch = np.zeros((3, 3), dtype=np.float32)
    dyOff, dxOff = _quadFit3x3(patch)
    assert dyOff == 0.0
    assert dxOff == 0.0


def testPeakAndSubpixel_integerPeakFound():
    corrMap = np.zeros((10, 10), dtype=np.float32)
    corrMap[4, 5] = 1.0
    truePeak, peakHeight = _peakAndSubpixel(corrMap, upsampleFactor=1)
    assert peakHeight == 1.0
    assert np.isclose(truePeak[0], 4.0, atol=1e-6)
    assert np.isclose(truePeak[1], 5.0, atol=1e-6)


def testPeakAndSubpixel_upsampleBranch():
    corrMap = np.zeros((10, 10), dtype=np.float32)
    corrMap[4, 5] = 1.0
    truePeak, peakHeight = _peakAndSubpixel(corrMap, upsampleFactor=2)
    assert peakHeight == 1.0
    assert truePeak.shape == (2,)


def testWraparoundPick_picksClosestToZero():
    shift = _wraparoundPick(np.array([8.0, 9.0]), (10, 10))
    # 8 - 10 = -2 (closer to 0 than 8), 9 - 10 = -1 (closer to 0 than 9)
    assert shift == (-2.0, -1.0)


def testWraparoundPick_inRangeStaysPut():
    shift = _wraparoundPick(np.array([2.0, -3.0]), (10, 10))
    assert shift == (2.0, -3.0)
