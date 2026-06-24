"""imageProcToolkit — image-processing toolkit used by the t2Interface L1->L2 pipeline.

A collection of flat accelerator / utility modules:

- ``fftUpsample``      — FFT zero-padding upsampler (cupy GPU backend, NumPy fallback).
- ``interp2``          — MATLAB ``interp2(...,'linear')`` port (numba fused-kernel backend).
- ``fftTranslateImage``— atomic FFT sub-pixel image translation.
- ``getTranslationalShifts`` — all-pairwise phase-correlation shift estimation (co-registration step 4).
- ``clampImageAmplitude``   — amplitude dynamic-range clamp (co-registration step 2).
- ``normalizeImageAmplitude``— per-image amplitude -> uint8 normalization (co-registration step 3).
- ``coTranslateImages``— the multi-image co-registration orchestrator (steps 2-5).

Import the public callables from their submodules explicitly, e.g.::

    from imageProcToolkit.fftUpsample import fourierUpsample
    from imageProcToolkit.interp2 import interp2linear

The L1->L2 pipeline modules (``geometricCorrectionPFA``, ``groundPlaneMatching``,
``processL1ToL2``, ``processPassL1ToL2``, ``t2CoordinateTransform``) live in a separate
consumer package and import this toolkit; they are not part of the package itself.
"""

__version__ = "0.1.0"