"""Tests for ``interp2`` (NumPy/CPU path).

See ``src/imageProcToolkit/interp2.py`` and the "Testing" section of
``CONVENTIONS.md``.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.interp2 import _interp2linear_numpy, interp2linear


def testInterp2Linear_interiorPointMatchesExactGrid():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    xi = np.array([1.0], dtype=np.float32)
    yi = np.array([1.0], dtype=np.float32)
    out = interp2linear(z, xi, yi)
    assert np.isclose(out[0], z[1, 1])


def testInterp2Linear_halfwayInterpolates():
    z = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    xi = np.array([0.5], dtype=np.float32)
    yi = np.array([0.5], dtype=np.float32)
    out = interp2linear(z, xi, yi)
    # average of the four corners: (0 + 2 + 4 + 6) / 4 = 3
    assert np.isclose(out[0], 3.0, atol=1e-5)


def testInterp2Linear_outOfRangeFillsExtrapval():
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    xi = np.array([5.0], dtype=np.float32)
    yi = np.array([0.0], dtype=np.float32)
    out = interp2linear(z, xi, yi, extrapval=np.nan)
    assert np.isnan(out[0])


def testInterp2Linear_farEdgeStepBack():
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    xi = np.array([2.0], dtype=np.float32)  # last column
    yi = np.array([0.0], dtype=np.float32)
    out = interp2linear(z, xi, yi)
    assert np.isclose(out[0], z[0, 2])


def testInterp2Linear_shapeMismatchRaises():
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    xi = np.array([0.0, 1.0], dtype=np.float32)
    yi = np.array([0.0], dtype=np.float32)
    try:
        interp2linear(z, xi, yi)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched xi/yi shapes")


def testInterp2Linear_smallZRaises():
    z = np.zeros((1, 1), dtype=np.float32)
    xi = np.array([0.0], dtype=np.float32)
    yi = np.array([0.0], dtype=np.float32)
    try:
        interp2linear(z, xi, yi)
    except ValueError:
        return
    raise AssertionError("expected ValueError for too-small z")


def testInterp2Linear_complex64Preserved():
    z = (
        (np.arange(16, dtype=np.float32) + 1j * np.arange(16, dtype=np.float32))
        .astype(np.complex64)
        .reshape(4, 4)
    )
    xi = np.array([1.5], dtype=np.float32)
    yi = np.array([1.5], dtype=np.float32)
    out = interp2linear(z, xi, yi)
    assert out.dtype == np.complex64


def testInterp2Linear_numpyReferenceOutofRange():
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    xi = np.array([-1.0, 1.5], dtype=np.float32)
    yi = np.array([0.0, 1.5], dtype=np.float32)
    out = _interp2linear_numpy(z, xi, yi, 3, 3, np.nan)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def testInterp2Linear_selfcheck_passes():
    from imageProcToolkit.interp2 import _interp2linear_selfcheck

    assert _interp2linear_selfcheck() is True
