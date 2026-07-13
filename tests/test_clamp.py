"""Tests for ``clamp`` (NumPy/CPU path).

See ``src/imageProcToolkit/clamp.py`` and the "Testing" section of
``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.clamp import clamp, histogramMode_dB, intensityPower_dB


def testIntensityPowerDB_returnsNegativeDecibels():
    intensity = np.array([[1.0, 0.1, 0.01]], dtype=np.float32)
    dB = intensityPower_dB(intensity)
    assert dB.shape == intensity.shape
    assert dB.dtype == np.float32
    assert np.isclose(dB[0, 0], 0.0)  # max -> 0 dB
    assert np.isclose(dB[0, 1], -10.0)  # 0.1 -> -10 dB
    assert np.isclose(dB[0, 2], -20.0)  # 0.01 -> -20 dB


def testIntensityPowerDB_allNanReturnsNan():
    intensity = np.full((3, 3), np.nan, dtype=np.float32)
    dB = intensityPower_dB(intensity)
    assert np.all(np.isnan(dB))


def testIntensityPowerDB_allNonPositiveReturnsNan():
    intensity = np.zeros((3, 3), dtype=np.float32)
    dB = intensityPower_dB(intensity)
    assert np.all(np.isnan(dB))


def testHistogramModeDB_returnsFiniteMode():
    rng = np.random.default_rng(0)
    dBarray = -np.abs(rng.standard_normal(1000)).astype(np.float32) * 30.0
    mode = histogramMode_dB(dBarray, outputDynamicRangePowerDB=60.0)
    assert np.isfinite(mode)
    assert mode <= 0.0


def testHistogramModeDB_noFiniteRaises():
    with np.testing.assert_raises(ValueError):
        histogramMode_dB(np.full(10, np.nan, dtype=np.float32))


def testHistogramModeDB_singleBinReturnsCenter():
    # inputDynamicRange small enough that only one bin is produced
    dBarray = np.array([-0.001, -0.002], dtype=np.float32)
    mode = histogramMode_dB(
        dBarray, outputDynamicRangePowerDB=0.1, inputDynamicRangePowerDB=0.01, binsPerDB=1
    )
    assert np.isfinite(mode)


def testHistogramModeDB_nonPositiveInputRangeFallback():
    # inputDynamicRangePowerDB <= 0 triggers the fallback to 1.0
    dBarray = np.array([-0.5, -1.0], dtype=np.float32)
    mode = histogramMode_dB(
        dBarray, outputDynamicRangePowerDB=60.0, inputDynamicRangePowerDB=-1.0, binsPerDB=5
    )
    assert np.isfinite(mode)


def testClamp_preservesShapeAndDtype():
    rng = np.random.default_rng(0)
    img = (rng.standard_normal((16, 16)) ** 2).astype(np.float32)
    out = clamp(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def testClamp_allNanReturnsNan():
    img = np.full((4, 4), np.nan, dtype=np.float32)
    out = clamp(img)
    assert out.shape == img.shape
    assert np.all(np.isnan(out))


def testClamp_jointAcrossChannels():
    rng = np.random.default_rng(1)
    base = (rng.standard_normal((16, 16)) ** 2).astype(np.float32)
    stack = np.stack([base, 0.1 * base, 0.01 * base], axis=-1)  # (16, 16, 3)
    joint = clamp(stack)
    flat = clamp(stack.reshape(-1, order="C")).reshape(stack.shape, order="C")
    assert np.array_equal(joint, flat, equal_nan=True)


def testClamp_selfcheck_passes():
    from imageProcToolkit.clamp import _clamp_selfcheck

    assert _clamp_selfcheck() is True
