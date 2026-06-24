# imageProcToolkit

Image-processing toolkit used by the `t2Interface` L1→L2 pipeline: FFT zero-padding
upsampling, MATLAB-style bilinear interpolation, atomic sub-pixel image translation,
all-pairwise phase-correlation shift estimation, amplitude clamping/normalization, and
the multi-image co-registration orchestrator.

## Installation

Editable install (for development):

```bash
pip install -e .
```

With the optional GPU fast-path (cupy; pick the wheel matching your CUDA toolkit):

```bash
pip install imageProcToolkit[gpu]
```

A usable GPU is **not** required — cupy is an optional accelerator with a NumPy fallback.
`numpy` and `numba` are installed automatically as required dependencies.

## Usage

Import the public callables from their submodules explicitly:

```python
from imageProcToolkit.fftUpsample import fourierUpsample
from imageProcToolkit.interp2 import interp2linear
from imageProcToolkit.fftTranslateImage import fftTranslateImage
from imageProcToolkit.getTranslationalShifts import getTranslationalShifts
from imageProcToolkit.clampImageAmplitude import clampImageAmplitude
from imageProcToolkit.normalizeImageAmplitude import normalizeImageAmplitude
from imageProcToolkit.coTranslateImages import coTranslateImages
```

## Modules

| Module | Description |
| --- | --- |
| `fftUpsample` | FFT zero-padding upsampler (cupy GPU backend, NumPy fallback). |
| `interp2` | MATLAB `interp2(...,'linear')` port (numba fused-kernel backend). |
| `fftTranslateImage` | Atomic FFT sub-pixel image translation. |
| `getTranslationalShifts` | All-pairwise phase-correlation shift estimation (co-registration step 4). |
| `clampImageAmplitude` | Amplitude dynamic-range clamp (co-registration step 2). |
| `normalizeImageAmplitude` | Per-image amplitude → uint8 normalization (co-registration step 3). |
| `coTranslateImages` | Multi-image co-registration orchestrator (steps 2-5). |

## License

MIT — see [LICENSE](LICENSE).