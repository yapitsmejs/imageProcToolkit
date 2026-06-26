"""imageProcToolkit — image-processing toolkit.

A collection of flat accelerator / utility modules:

- ``fftUpsample``      — FFT zero-padding upsampler (cupy GPU backend, NumPy fallback).
- ``interp2``          — MATLAB ``interp2(...,'linear')`` port (numba fused-kernel backend).
- ``fftTranslate2d``— atomic FFT sub-pixel image translation.
- ``getTranslationalShifts`` — all-pairwise phase-correlation shift estimation (co-registration step 4).
- ``clampAmplitude``   — amplitude dynamic-range clamp (co-registration step 2).
- ``normalizeArray``— per-image amplitude -> uint8 normalization (co-registration step 3).
- ``coTranslate2d``— the multi-image translation co-registration orchestrator (steps 2-5).
- ``getSimilarityTransform`` — all-pairwise Fourier-Mellin rotation/scale + global similarity (step 4b; 4-DOF generalization of getTranslationalShifts).
- ``similarityTransform2d``— atomic 4-DOF similarity warp (rotation + uniform scale + translation).
- ``coSimilarityTransform2d``— multi-image similarity co-registration orchestrator (steps 2-3-4b-5).

Import the public callables from their submodules explicitly, e.g.::

    from imageProcToolkit.fftUpsample import fourierUpsample
    from imageProcToolkit.interp2 import interp2linear
"""

__version__ = "0.1.0"