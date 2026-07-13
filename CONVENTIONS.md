# Conventions

This document is the single source of truth for naming, code style, and the
domain-specific patterns used to maintain this project. It is intentionally
tool-agnostic — it does not assume any particular editor, linter UI, or AI
assistant. `CLAUDE.md` (when present) points here and holds only agent-memory
aid, not conventions.

## Naming conventions

This project uses **camelCase for the name and a snake_case underscore as a
separator for different aspects** of an identifier. The underscore-separated tail
carries a unit, coordinate frame, or other aspect; the camelCase part carries
the quantity's name.

- **Variables:** `camelCase` + `_aspect` suffix.
  - Units: `heightOfX_m`, `targetBeamAngle_rad`, `radarCenterFrequency_Hz`,
    `aperturePowerBound_dB`, `fastTimeAxis_s`, `wavePropogationSpeed_mPerSec`,
    `pulseRepFreq_Hz`.
  - Coordinate frames as a composite suffix: `apcStartPos_XYZ_m`,
    `apcVelVec_XYZ_mPerSec`, `sceneCenter_ENU_m`.
  - Unitless quantities are plain camelCase: `pixelWiseDelay`, `interpValues`,
    `phaseCorrection`.
- **Modules / files & the import package:** camelCase — `normalizeSignal.py`,
  `computeStats.py`, the package `imageProcToolkit/`. Private modules lead with
  an underscore: `_chunking.py`, `_phaseCorrelationCore.py`.
- **Functions:** camelCase public — `formImage`, `computeFbr`, `getTxPos`.
  Private helpers are `_camelCase` — `_robustMode`, `_selfcheck`. Backend-dispatch
  compute cores use the `_<func>_core(..., xp)` pattern (one code path for
  numpy/cupy, no mirrored `_<func>_cupy` duplicates): `_fourierUpsample_core`,
  `_fftTranslate_core`, `_clamp_core`.
- **Classes:** PascalCase — `Transaction`, `Statement`, `Level`.
- **Constants:** UPPER_SNAKE. Private constants keep a single leading underscore
  — `_HAVE_CUPY_GPU`, `_HAVE_NUMBA`, `_SKIP`.

## Code rigor

- Start every module with `from __future__ import annotations` (deferred
  evaluation, enables PEP 604 on older targets).
- **Type hints are required** on public functions. Use PEP 604/585 style:
  `x | None`, `list[int]`, `dict[str, float]`, `tuple[float, float] | None`. Do
  not use `typing.Optional` / `typing.List`. Enforced by **mypy**
  (`uv run mypy src/imageProcToolkit`, and via the gate) — the gate fails on
  an untyped public function.
- **Group imports** in order: future → stdlib → third-party → local, with a
  blank line between groups. Let `ruff` (rule `I`) own the ordering.
- **Google-style docstrings** on every public function:
  ```python
  def clamp(array, outputDynamicRangePowerDB=60.0, inputDynamicRangePowerDB=None):
      """N-D intensity array -> clamped linear intensity array (same shape).

      Args:
          array: real-valued non-negative intensity array.
          outputDynamicRangePowerDB: desired output dynamic range in dB.

      Returns:
          float32 clamped linear intensity, same shape as ``array``.
      """
  ```
- Declare the public API with `__all__` in `__init__.py` and in tool modules
  that export helpers.
- **ruff** selects `E, F, W, I, UP, B` and ignores `E501`. `N` (pep8-naming) is
  deliberately excluded so camelCase identifiers are not flagged — do not add it.

## Testing / Test-driven development

This project is test-driven. Features land with tests, and a coverage gate
keeps it that way.

- **Test-first (red-green-refactor):** before writing or changing behavior,
  write or extend the relevant `tests/test_<module>.py` test and run it red
  (`uv run pytest tests/test_<module>.py`). Then write the minimal
  implementation to turn it green, then refactor.
- **Dev loop:** `uv run pytest` runs the full suite fast, with **no coverage
  enforcement** — use it during red/green work.
- **Gate:** `uv run python scripts/check.py` runs ruff check, ruff format
  `--check`, **mypy**, and pytest with branch coverage. Branch coverage on the
  package must stay **≥ 90%**. Run it (and it must pass) before pushing or opening
  a PR. CI (`.github/workflows/ci.yml`) runs the same gate on every push and PR,
  so a red PR cannot merge.
- **Test naming:** `test` + camelCase name, optionally a `_aspect` suffix —
  `testVersionIsSet`, `testClampValue_clampsBelowLow`. (Same camelCase rule as
  the rest of the codebase; `N`/pep8-naming stays excluded in ruff.)
- **Layout:** one `tests/test_<module>.py` per source module
  (`clamp.py` → `tests/test_clamp.py`). `test_smoke.py` is the lone
  cross-cutting smoke test.

## Commit convention

Use conventional commit types: `feat`, `fix`, `docs`, `style`, `refactor`,
`perf`, `test`, `build`, `ci`, `chore`, `revert`.

## CuPy / dual CPU-GPU backend convention

This package follows the cross-repo CuPy array-module convention (canonical
reference: `sarChangeDetection/CONVENTIONS.md`). The rule: one code path
parameterized by the array module, with CuPy as an optional, on-demand
dependency.

- `cupy` is **never** a declared runtime dependency — the correct wheel depends
  on the host GPU/CUDA and cannot be chosen statically (see
  `scripts/install_cupy.py`). It is imported eagerly at module load under
  `try/except`; `_HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0` decides
  at import time (CPU-only hosts fall back to NumPy, never hard-fail). A
  *usable* CUDA device is required, not just an importable cupy.
- Compute helpers are `xp`-parameterized (`_<func>_core(..., xp)`): the same code
  runs on numpy or cupy — no `if gpu:`/`else:` mirror branches, no
  `_<func>_cupy` duplicate. Used in `_phaseCorrelation_core`,
  `_fourierUpsample_core`, `_fftTranslate_core`.
- The public function auto-dispatches on `if _HAVE_CUPY_GPU:` — **cupy in →
  cupy out** (kept resident, detected with `isinstance(x, cp.ndarray)`) and
  **numpy in → numpy out** (auto round-tripped via `cp.asnumpy`). The legacy
  return-gpu opt-out is gone: a caller that wants to keep arrays on device passes
  cupy arrays through.
- Memory cleanup, where used, is the public `cp.get_default_memory_pool()
  .free_all_blocks()` (never the private `cp._default_memory_pool`).
- `cp` is typed `Any` because cupy mirrors numpy's API but is not stub-typed; the
  array-module dispatch (`xp` param) is likewise `Any`. Functions returning an
  `xp.something` (which is `Any`) assign to a `result: np.ndarray = ...` typed
  local before returning (avoids mypy `warn_return_any`).
- Each array-module file ships a `_<func>_cupy_selfcheck()` (cupy path vs a
  NumPy reference produced by calling `_core(..., np)` directly, bypassing
  auto-dispatch), run from `if __name__ == "__main__":` and gated on
  `_HAVE_CUPY_GPU` (`skipped (no GPU available …)` otherwise). Run them
  directly, e.g. `uv run python -m imageProcToolkit.fftUpsample`.

### GPU coverage rule

The gate (`scripts/check.py`) runs on CPU CI (no GPU), so the GPU dispatch
branch (`if _HAVE_CUPY_GPU:`), the cupy import try-body, the
`_<func>_cupy_selfcheck()` GPU body, and the `__main__` selfcheck entry are marked
`# pragma: no cover` to keep them out of the coverage denominator. **The ≥90%
gate therefore applies to the CPU-reachable (NumPy) path only.** This is the
documented accommodation for GPU-only code, not a way to skip real coverage —
the NumPy path must be fully tested. A future GPU CI job would lift the pragma.

## Co-registration pipeline (5-step recipe shared by two orchestrators)

`coTranslate2d` (2-DOF translation) and `coSimilarityTransform2d` (4-DOF
similarity = rotation + uniform scale + translation) run the same conceptual
pipeline. Reading either orchestrator requires understanding all five steps,
which span multiple modules:

1. **toIntensity** (`_toIntensity` in each orchestrator) — resolve inputs to
   real float32 intensity per `arrayScale`: complex → `|z|²`, real `'amplitude'`
   → `x²`, real `'intensity'` → passthrough.
2. **clamp** (`clamp.clamp`) — 10·log10 intensity-dB dynamic-range clamp. `clamp`
   is unit-unaware (speaks only intensity), so the orchestrator resolves to
   intensity first.
3. **normalize** (`normalizeArray.normalizeToUint8`) — clamped intensity →
   per-image uint8.
4. **estimate** (`getTranslationalShifts` / `getSimilarityTransform`) —
   all-pairwise estimation on the uint8, then a global least-squares solve for
   per-image params.
5. **apply** (`fftTranslate2d` / `similarityTransform2d`) — warp each **original
   input** by its per-image params. dtype-preserving: complex in → complex out
   (phase preserved), real in → real out.

**The key branch / invariant:** steps 2–4 are *estimation-only* on the
uint8/clamped-intensity proxy (phase correlation and Fourier-Mellin are
brightness-invariant). Step 5 applies the estimated transform to the
**original inputs** — this is what preserves complex phase. `arrayScale` is the
input-unit contract threaded from step 2 through step 5; the warps are
unit-preserving.

## Estimation = pairwise + a gauge-constrained global solve

Both estimators follow the same shape: compute pairwise transforms for a set of
image pairs, then solve a global least-squares for per-image params. The normal
equations are `L p = -d` where `L` is the unweighted graph Laplacian of the
observation graph and `d` aggregates the pairwise edge vectors. `L` is singular
(the all-ones null space = the gauge freedom), so a gauge must be fixed:

- **zero-mean gauge** (default, `masterIndex=None`): all `n(n-1)/2` pairs are
  estimated (the complete graph), then `sum(p) = 0` via the augmented system
  `[[L, 1],[1.T, 0]] @ [p, lam] = [-d, 0]`. No image is ground truth; the
  correction is distributed symmetrically. O(N²) estimation.
- **fix-node gauge** (`masterIndex=k`, negative wraps so `-1` = last image): pin
  image `k` at the identity (`p_k = 0`) by dropping row/col `k` and solving the
  reduced full-rank `L_ff p_f = -d_f`. **Setting `masterIndex` also switches
  estimation to the O(N) star graph** — only the `n-1` master↔image pairs are
  estimated, not all pairs. The orchestrators build
  `starPairs = [(min(i, k), max(i, k)) for i in range(n) if i != k]` and pass it
  to the estimators via their `pairs=` filter. The star + fix-node solve is exact:
  one edge per non-master image ⇒ `p_i` = "transform to apply to i to align it to
  master k". Trade-off: no cross-checking between non-master images, so less
  noise-robust than all-pairs.

The shared gauge solve lives in `getTranslationalShifts._solveLaplacianGauge(L,
d, masterIndex)` and is reused by
`getSimilarityTransform.solveGlobalSimilarityRotScale`. The `pairs=None` filter
on `allPairwiseTranslationalShifts` / `allPairwiseSimilarityTransforms` is the
mechanism that bounds estimation scope; `None` = all pairs. Both orchestrators
thread `masterIndex` through; `coSimilarityTransform2d` uses the **same** master
(and the same star pair list) for both its rot/scale stage and translation stage
so the composed `(theta, log s, dy, dx)` pins the master end-to-end.

### Sign convention (critical, easy to get wrong)

`pairwise...(A, B)` returns the transform to **apply to B** to align it to
**A** (A = master, B = slave). The per-image params satisfy
`p_j - p_i = s_ij` at the optimum. For translation `s_ji = -s_ij`; for rot/scale
`theta_ji = -theta_ij`, `log s_ji = -log s_ij`. The scale column in the params is
**log s** — `exp` it for the applied scale (the warper does this internally). The
`(min, max)` edge keying in the star pair list preserves this convention, so the
existing solve handles the star with no sign special-casing.

## `coSimilarityTransform2d` is a two-stage decoupled solve

Stage 1 estimates rotation+scale via Fourier-Mellin (log-polar spectra + phase
correlation) on the uint8 magnitude spectra → per-image `(theta, log s)`. Stage
2 de-rotates + de-scales the **clamped intensity** (float32, not the uint8),
re-derives masks from `np.isfinite` of the de-warped float (the warp NaN-marks
its out-of-source border), re-normalizes to uint8, and reuses
`getTranslationalShifts` for `(dy, dx)`. This decoupling exists because
rotation/scale and translation are solved in different coordinate systems
(Lie-algebra-additive rot/scale vs. additive translation). **Consequence:** the
two stages are not independently gauge-equivalent — stage-2 inputs depend on
stage-1's gauge via the dewarp, so re-running with a different `masterIndex`
perturbs stage 2 slightly. With the star-estimation change, the master run also
uses a *different set of pairwise measurements* than the all-pairs zero-mean
run, so `params_master ≠ params_zm − params_zm[master]` even for
`coTranslate2d` — do not assert gauge-equivalence across the two modes.

## Masks convention

**Masks come from the input, never the uint8.** `normalizeToUint8` maps NaN → 0
sentinel, so NaN borders are undetectable in the uint8. Derive valid-pixel masks
from `np.isfinite` of the original input (or of the de-warped float, in stage 2).

## Numerics that are easy to break

- **Dtype preservation**: `np.fft.fft2` upcasts complex64 → complex128 (and
  float32 → complex128); cast back with `.astype(image.dtype)` at a single
  return point so the on-disk complex64 row-major contract is preserved.
- **Real-input sub-pixel caveat** (from `fftTranslate2d`, carried through the
  similarity warp): the real-input translate branch takes `np.abs` of the
  phase-ramp IFFT, which is only faithful for integer shifts; sub-pixel
  translations of a *real* input have minor distortion from the `np.abs` fold.
  The complex branch (the interferometry case) has no such fold.
- **NaN borders grow on every look**: the warp marks out-of-source pixels NaN
  (interp2 `extrapval=NaN`); `fftTranslate2d` zero-fills input NaN before its FFT
  (an NaN anywhere in `fft2` poisons the whole array) and, with
  `markInvalid=True` (default), NaN-fills the shifted-out border. So a transform
  of an image with a NaN border grows that border — the intended, cleaner
  behaviour for interferometry.
- **Crop-slice (zero-shift no-op)**: the padded array is cropped back with
  `[padY:padY+H, padX:padX+W]`. The naively symmetric `[padY:-padY, padX:-padX]`
  would collapse to the empty slice `[0:0]` when a pad component is 0 (zero
  shift); the explicit end-index form is a true no-op slice when pad = 0.

## Extensibility

- New accelerators should be flat modules under `src/imageProcToolkit/` (no
  subpackages). Keep public functions camelCase.
- Runtime deps stay limited to `numpy`, `numba`. CuPy must never be added to
  `[project.dependencies]`; it is installed on demand via
  `scripts/install_cupy.py`.