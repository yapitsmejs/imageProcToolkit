from __future__ import annotations

import imageProcToolkit


def testVersionIsSet():
    assert isinstance(imageProcToolkit.__version__, str)
    assert imageProcToolkit.__version__
