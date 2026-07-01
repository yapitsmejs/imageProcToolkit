# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`imageProcToolkit` — a flat, src-layout Python package of image-processing accelerators and a multi-image co-registration pipeline. Complex (SAR / interferometry) inputs are first-class: phase is preserved through co-registration. `numpy` + `numba` are required; `cupy` is an optional GPU fast-path with a NumPy fallback. The package is **not on PyPI** — it is installed from GitHub.

## Commands

The project uses `uv` (there is a `uv.lock`). There is no declared test dependency, so pytest is pulled ephemerally:

```bash
uv sync                                    # install numpy/numba into .venv
uv run --with pytest pytest tests/ -q      # run the smoke tests
uv run --with pytest pytest tests/test_smoke.py::test_coTranslate2d_master_index_pins_last -q   # one test
```

If the network is unavailable, the smoke tests are plain `test_*` functions with bare asserts and no pytest import, so they also run under a trivial manual harness:
```bash
uv run python -c "import tests.test_smoke as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
```

Every module that does estimation or warping ships a `_<moduleName>_selfcheck()` function that synthesizes a known transform and verifies recovery — these are the real correctness arbiter, not the smoke tests. Run the one for the module you touched:
```bash
uv run python -c "from imageProcToolkit.coSimilarityTransform2d import _coSimilarityTransform2d_selfcheck as f; f()"
```
Self-checks print `... self-check: PASS (...)`. There is no linter/formatter configured.

`scripts/install_cupy.py` auto-detects GPU + CUDA Toolkit and installs the matching `cupy-cuda{MAJOR}x` wheel (or `[ctk]` bundled variant, or nothing if no GPU) into `.venv`. It is run from a repo root and auto-detects `REPO_ROOT/.venv`.

## Architecture

### The co-registration pipeline is a 5-step recipe shared by two orchestrators

`coTranslate2d` (2-DOF translation) and `coSimilarityTransform2d` (4-DOF similarity = rotation + uniform scale + translation) run the same conceptual pipeline. Reading either orchestrator requires understanding all five steps, which span multiple modules:

1. **toIntensity** (`_toIntensity` in each orchestrator) — resolve inputs to real float32 intensity per `arrayScale`: complex → `|z|²`, real `'amplitude'` → `x²`, real `'intensity'` → passthrough.
2. **clamp** (`clamp.clamp`) — 10·log10 intensity-dB dynamic-range clamp. `clamp` is unit-unaware (speaks only intensity), so the orchestrator resolves to intensity first.
3. **normalize** (`normalizeArray.normalizeToUint8`) — clamped intensity → per-image uint8.
4. **estimate** (`getTranslationalShifts` / `getSimilarityTransform`) — all-pairwise estimation on the uint8, then a global least-squares solve for per-image params.
5. **apply** (`fftTranslate2d` / `similarityTransform2d`) — warp each **original input** by its per-image params. dtype-preserving: complex in → complex out (phase preserved), real in → real out.

**The key branch / invariant:** steps 2–4 are *estimation-only* on the uint8/clamped-intensity proxy (phase correlation and Fourier-Mellin are brightness-invariant). Step 5 applies the estimated transform to the **original inputs** — this is what preserves complex phase. `arrayScale` is the input-unit contract threaded from step 2 through step 5; the warps are unit-preserving.

### Estimation = pairwise + a gauge-constrained global solve

Both estimators follow the same shape: compute pairwise transforms for a set of image pairs, then solve a global least-squares for per-image params. The normal equations are `L p = -d` where `L` is the unweighted graph Laplacian of the observation graph and `d` aggregates the pairwise edge vectors. `L` is singular (the all-ones null space = the gauge freedom), so a gauge must be fixed:

- **zero-mean gauge** (default, `masterIndex=None`): all `n(n-1)/2` pairs are estimated (the complete graph), then `sum(p) = 0` via the augmented system `[[L, 1],[1.T, 0]] @ [p, lam] = [-d, 0]`. No image is ground truth; the correction is distributed symmetrically. O(N²) estimation.
- **fix-node gauge** (`masterIndex=k`, negative wraps so `-1` = last image): pin image `k` at the identity (`p_k = 0`) by dropping row/col `k` and solving the reduced full-rank `L_ff p_f = -d_f`. **Setting `masterIndex` also switches estimation to the O(N) star graph** — only the `n-1` master↔image pairs are estimated, not all pairs. The orchestrators build `starPairs = [(min(i, k), max(i, k)) for i in range(n) if i != k]` and pass it to the estimators via their `pairs=` filter. The star + fix-node solve is exact: one edge per non-master image ⇒ `p_i = "transform to apply to i to align it to master k"`. Trade-off: no cross-checking between non-master images, so less noise-robust than all-pairs.

The shared gauge solve lives in `getTranslationalShifts._solveLaplacianGauge(L, d, masterIndex)` and is reused by `getSimilarityTransform.solveGlobalSimilarityRotScale` (`getSimilarityTransform` imports `getTranslationalShifts`). The `pairs=None` filter on `allPairwiseTranslationalShifts` / `allPairwiseSimilarityTransforms` is the mechanism that bounds estimation scope; `None` = all pairs. Both orchestrators thread `masterIndex` through; `coSimilarityTransform2d` uses the **same** master (and the same star pair list) for both its rot/scale stage and translation stage so the composed `(theta, log s, dy, dx)` pins the master end-to-end.

**Sign convention (critical, easy to get wrong):** `pairwise...(A, B)` returns the transform to **apply to B** to align it to **A** (A = master, B = slave). The per-image params satisfy `p_j - p_i = s_ij` at the optimum. For translation `s_ji = -s_ij`; for rot/scale `theta_ji = -theta_ij`, `log s_ji = -log s_ij`. The scale column in the params is **log s** — `exp` it for the applied scale (the warper does this internally). The `(min, max)` edge keying in the star pair list preserves this convention, so the existing solve handles the star with no sign special-casing.

### `coSimilarityTransform2d` is a two-stage decoupled solve

Stage 1 estimates rotation+scale via Fourier-Mellin (log-polar spectra + phase correlation) on the uint8 magnitude spectra → per-image `(theta, log s)`. Stage 2 de-rotates + de-scales the **clamped intensity** (float32, not the uint8), re-derives masks from `np.isfinite` of the de-warped float (the warp NaN-marks its out-of-source border), re-normalizes to uint8, and reuses `getTranslationalShifts` for `(dy, dx)`. This decoupling exists because rotation/scale and translation are solved in different coordinate systems (Lie-algebra-additive rot/scale vs. additive translation). **Consequence:** the two stages are not independently gauge-equivalent — stage-2 inputs depend on stage-1's gauge via the dewarp, so re-running with a different `masterIndex` perturbs stage 2 slightly. With the star-estimation change, the master run also uses a *different set of pairwise measurements* than the all-pairs zero-mean run, so `params_master ≠ params_zm − params_zm[master]` even for `coTranslate2d` — do not assert gauge-equivalence across the two modes.

### Module roles and dependency shape

The package is **flat** (no subpackages). Dependency flow is roughly:

```
fftUpsample ──┐
              ├─> _phaseCorrelationCore ──> getTranslationalShifts ──┐
interp2 ──────┘                                      │              ├─> coTranslate2d
   │                                                ┌─┘              │
   └────> similarityTransform2d ────────────────────┤                ├─> coSimilarityTransform2d
                                       fftTranslate2d ───────────────┘
clamp, normalizeArray — leaf utilities used by the orchestrators
```

- **`_phaseCorrelationCore`** — the sign-sensitive phase-correlation primitives (normalized cross-power surface, 3×3 quad-fit sub-pixel peak, wraparound pick) shared by both estimators. It also owns the **cupy GPU gate**: `_HAVE_CUPY_GPU` is detected at import (a CUDA device must actually be present; cupy importing is not enough) and the NumPy/`cp` dispatch is internal to `_phaseCorrelationMap`. The two estimator modules re-bind these private names at import time (`from ._phaseCorrelationCore import ...`) so their public surface — including private names callers may have imported — is unchanged. Do not reference `cp` / `_HAVE_CUPY_GPU` from the estimator modules; keep the dispatch inside `_phaseCorrelationCore`.
- **`fftUpsample`** / **`interp2`** — accelerators with backends: cupy GPU (with NumPy fallback) and numba fused-kernel respectively. `interp2` is a MATLAB `interp2(...,'linear')` port.
- **`fftTranslate2d`** — atomic FFT sub-pixel translation; **`similarityTransform2d`** — atomic 4-DOF similarity warp (bilinear inverse-map rot/scale, then FFT phase-ramp translate). `similarityTransformImages` is the batched loop over `similarityTransform2d`; `coTranslate2d` defines its own `fftTranslateImages` batched loop.

### Conventions to follow

- **Masks come from the input, never the uint8.** `normalizeToUint8` maps NaN → 0 sentinel, so NaN borders are undetectable in the uint8. Derive valid-pixel masks from `np.isfinite` of the original input (or of the de-warped float, in stage 2).
- **Module-level docstrings are the design doc.** Each estimator/orchestrator module opens with a long `'''...'''` docstring explaining the math (sign convention, gauge, residual model), the step decomposition, and the rationale. Match this style and keep these in sync when changing behavior — they are where the "why" lives (the `'''` string is the module docstring, not a comment).
- **Self-checks are the arbiter.** When changing an estimator or warper, update/run the module's `_<moduleName>_selfcheck()` and confirm `PASS` with the recovered transform matching the synthetic ground truth to the documented tolerance.
- **Public callables are imported from their submodules explicitly** (`from imageProcToolkit.X import Y`), not from the package root — keep the package `__init__` thin (it only carries `__version__` and a docstring).