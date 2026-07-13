"""Tests for ``fftUpsample`` (NumPy/CPU path).

See ``src/imageProcToolkit/fftUpsample.py`` and the "Testing" section of
``CONVENTIONS.md``. The CuPy/GPU dispatch branch is excluded from coverage
(``# pragma: no cover``) -- it cannot run on CPU CI.
"""

from __future__ import annotations

import numpy as np

from imageProcToolkit.fftUpsample import _fourierUpsample_core, fourierUpsample


def testFourierUpsample_doublesLengthAlongAxis():
    a = np.zeros((8, 8), dtype=np.float32)
    a[4, 4] = 1.0
    out = fourierUpsample(a, up=2, axis=0)
    assert out.shape == (16, 8)


def testFourierUpsample_preservesComplex64():
    a = (np.arange(16) + 1j * np.arange(16)).astype(np.complex64).reshape(4, 4)
    out = fourierUpsample(a, up=2, axis=1)
    assert out.shape == (4, 8)
    assert out.dtype == np.complex64


def testFourierUpsample_coreNumpyPath():
    a = np.ones((4, 4), dtype=np.float32)
    out = _fourierUpsample_core(a, 2, 0, np)
    assert out.shape == (8, 4)


def testFourierUpsample_cupySelfcheck_runsOnCpuAndReportsSkip(capsys):
    from imageProcToolkit.fftUpsample import _fourierUpsample_cupy_selfcheck

    assert _fourierUpsample_cupy_selfcheck() is True
    captured = capsys.readouterr()
    assert "skipped" in captured.out  # CPU host: GPU path reported as skipped
