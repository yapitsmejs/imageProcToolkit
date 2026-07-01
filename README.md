# imageProcToolkit

Image-processing toolkit: FFT zero-padding upsampling, MATLAB-style bilinear
interpolation, atomic sub-pixel image translation, all-pairwise phase-correlation
shift estimation, intensity clamping/normalization, and multi-image co-registration
(translation and 4-DOF similarity). Complex (SAR / interferometry) inputs are
first-class — phase is preserved through the co-registration pipeline. `numpy` and
`numba` are required; `cupy` is an **optional** GPU fast-path with a NumPy fallback
when it (or a CUDA device) is absent.

## Install — pull from GitHub into an external repo

The package is **not on PyPI**; install it directly from GitHub. `numpy` and `numba`
are pulled automatically as required dependencies. `cupy` is intentionally **not**
installed here — it is a machine-specific CUDA wheel (see
[Set up cupy](#set-up-cupy-optional-gpu-fast-path) below).

### uv (this project's own toolchain)

New project:

```bash
uv init myrepo && cd myrepo
uv add "git+https://github.com/yapitsmejs/imageProcToolkit.git"
```

Pin to a specific commit (or a tag, once one is cut — same `@<ref>` syntax):

```bash
uv add "git+https://github.com/yapitsmejs/imageProcToolkit.git@<commit-sha>"
```

### pip (any virtualenv)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      | macOS/Linux:  source .venv/bin/activate
python -m pip install "git+https://github.com/yapitsmejs/imageProcToolkit.git"
```

Pin to a commit/tag:

```bash
python -m pip install "git+https://github.com/yapitsmejs/imageProcToolkit.git@<commit-sha>"
```

## Set up cupy (optional GPU fast-path)

A usable GPU is **not** required — the toolkit runs on the NumPy fallback when
`cupy` is absent or no CUDA device is detected. Install cupy only if you want the
GPU path, and install it into the **same venv** that holds `imageProcToolkit`.

### Option A — manual (recommended, self-contained)

1. Detect your CUDA major version:

   ```bash
   nvcc --version        # CUDA Toolkit present -> "release X.Y" gives the major
   nvidia-smi           # no toolkit -> read the "CUDA Version:" the driver supports
   ```

2. Install the matching wheel. Pick `cupy-cuda11x`, `cupy-cuda12x`, or `cupy-cuda13x`
   by your CUDA major (11, 12, or 13). If you have a GPU but **no** CUDA Toolkit, add
   the `[ctk]` extra so the wheel bundles CUDA libraries via PyPI.

   uv:

   ```bash
   uv pip install cupy-cuda13x                       # toolkit present -> system CUDA
   uv pip install "cupy-cuda12x[ctk]"                # GPU, no toolkit -> bundled CUDA
   ```

   pip:

   ```bash
   python -m pip install cupy-cuda13x
   python -m pip install "cupy-cuda12x[ctk]"
   ```

   Only **one** `cupy-cuda*x` distribution may be installed at a time — if you are
   upgrading or switching CUDA versions, uninstall the others first:

   ```bash
   uv pip uninstall cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x
   # or:  python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x
   ```

3. Verify a device is visible:

   ```bash
   python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.getDeviceCount())"
   ```

### Option B — reuse the toolkit's auto-detecting installer (convenience)

This repo ships `scripts/install_cupy.py`, which auto-detects the GPU + CUDA Toolkit
and installs the right wheel (no GPU → installs nothing; GPU + toolkit →
`cupy-cuda{MAJOR}x`; GPU, no toolkit → `cupy-cuda{MAJOR}x[ctk]`). It installs into its
**own** repo's `.venv`, so to use it for your external repo, fetch just the script and
run it from your repo root (it auto-detects `REPO_ROOT/.venv`):

```bash
# macOS/Linux
curl -fsSLO https://raw.githubusercontent.com/yapitsmejs/imageProcToolkit/main/scripts/install_cupy.py
python install_cupy.py

# Windows PowerShell
Invoke-WebRequest -UseBasicParsing -OutFile install_cupy.py `
  https://raw.githubusercontent.com/yapitsmejs/imageProcToolkit/main/scripts/install_cupy.py
python install_cupy.py
```

(Alternatively, `git clone` the toolkit and run `uv run python scripts/install_cupy.py`
inside it, then point that venv at your project.)

## Usage

Import the public callables from their submodules explicitly:

```python
import numpy as np
from imageProcToolkit.fftUpsample import fourierUpsample
from imageProcToolkit.interp2 import interp2linear
from imageProcToolkit.fftTranslate2d import fftTranslate2d
from imageProcToolkit.getTranslationalShifts import getTranslationalShifts
from imageProcToolkit.clamp import clamp
from imageProcToolkit.normalizeArray import normalizeToUint8
from imageProcToolkit.coTranslate2d import coTranslate2d
from imageProcToolkit.getSimilarityTransform import getSimilarityTransform
from imageProcToolkit.similarityTransform2d import similarityTransform2d
from imageProcToolkit.coSimilarityTransform2d import coSimilarityTransform2d
```

Co-register a stack of complex or real images — `arrayScale` declares the inputs' unit
(`'amplitude'` or `'intensity'`). The transform is estimated on an intensity-derived
proxy and applied to the **original** inputs, so complex in → complex out (phase
preserved for interferometry), real in → real out:

```python
imgs = [np.ndarray, ...]                       # complex or real, all same shape
transformed, params, diag = coTranslate2d(imgs, arrayScale='amplitude')
# transformed : list of N arrays, co-registered (same dtype/unit as the inputs)
# params      : (N, 2) per-image shifts (dy, dx). Default gauge is zero-mean
#               (sum(shifts) = 0, no image is ground truth).
# diag        : diagnostics dict (nPairs, residuals)
```

For rotation/scale + translation co-registration (inputs need not be rotation-aligned),
use `coSimilarityTransform2d` instead — it returns `(transformed, params(N,4), diag)`
with per-image `(theta, s, dy, dx)`.

### Register toward a master image

By default both orchestrators estimate **every** image pair (O(N²) pairs) and solve a
zero-mean gauge (no image is ground truth). Pass `masterIndex=k` (negative wraps, so
`-1` = the last image) to pin image `k` at the identity and register every other image
toward it; this also switches estimation to the **O(N)** star graph (only the N−1
master↔image pairs are estimated), which is faster for large stacks:

```python
transformed, params, diag = coTranslate2d(imgs, arrayScale='amplitude', masterIndex=-1)
# params[masterIndex] is exactly the identity (0,0) -- or (0,0,0,0) for the
# similarity variant. Every other row is the transform that aligns that image
# to the master. diag['nPairs'] == N-1 (star), not N(N-1)/2 (all pairs).
```

Trade-off: the star uses one measurement per non-master image with no cross-checking
from other pairs, so it is less noise-robust than the all-pairs least-squares.

## Modules

| Module | Description |
| --- | --- |
| `fftUpsample` | FFT zero-padding upsampler (cupy GPU backend, NumPy fallback). |
| `interp2` | MATLAB `interp2(...,'linear')` port (numba fused-kernel backend). |
| `fftTranslate2d` | Atomic FFT sub-pixel image translation. |
| `getTranslationalShifts` | All-pairwise phase-correlation shift estimation (co-registration step 4). |
| `clamp` | Intensity dynamic-range clamp (10·log10; co-registration step 2). |
| `normalizeArray` | Per-image amplitude → uint8 normalization (co-registration step 3). |
| `coTranslate2d` | Multi-image translation co-registration orchestrator (steps 2-5). |
| `getSimilarityTransform` | All-pairwise Fourier-Mellin rotation/scale + global similarity (step 4b). |
| `similarityTransform2d` | Atomic 4-DOF similarity warp (rotation + uniform scale + translation). |
| `coSimilarityTransform2d` | Multi-image similarity co-registration orchestrator (steps 2-3-4b-5). |

## License

MIT — see [LICENSE](LICENSE).