# CLAUDE.md

This file provides agent-memory aid for Claude Code (claude.ai/code) when
working with code in this repository. **Naming, code style, the CuPy/GPU
convention, the 5-step co-registration pipeline, sign convention, gauge rules,
and the commit convention live in [`CONVENTIONS.md`](CONVENTIONS.md)** — read
that first and follow it. This file only holds project-specific context that
does not belong in the conventions doc.

## Project

`imageProcToolkit` — a flat, src-layout Python package of image-processing
accelerators and a multi-image co-registration pipeline. Complex (SAR /
interferometry) inputs are first-class: phase is preserved through
co-registration. `numpy` + `numba` are required; `cupy` is an optional GPU
fast-path with a NumPy fallback. The package is **not on PyPI** — it is installed
from GitHub. Import name `imageProcToolkit` (camelCase), dist name
`image-proc-toolkit` (kebab).

## Commands

This project is [uv](https://docs.astral.sh/uv/)-managed.

```bash
uv sync --extra dev                                          # .venv + runtime deps (numpy, numba) + dev tooling (pytest, ruff, mypy)
uv run python scripts/check.py                               # GATE: ruff + format + mypy + tests + >=90% branch coverage
uv run pytest                                                # dev loop (no coverage gate)
uv run python -m imageProcToolkit.fftUpsample                 # __main__ self-check (cupy-vs-NumPy; skips GPU part on CPU hosts)
uv run python -m imageProcToolkit._phaseCorrelationCore        # __main__ self-check (cupy-vs-NumPy; skips GPU part on CPU hosts)
uv run python -m imageProcToolkit.clamp                        # __main__ self-check (clamp joint-across-channels)
uv run python -m imageProcToolkit.interp2                     # __main__ self-check (numba-vs-NumPy; skips on no-numba)
uv run python -m imageProcToolkit.similarityTransform2d        # __main__ self-check (rot/scale round-trip + dtype preservation)
uv run python -m imageProcToolkit.getSimilarityTransform       # __main__ self-check (2-node known-similarity recovery, the sign arbiter)
uv run python -m imageProcToolkit.coSimilarityTransform2d     # __main__ self-check (2-image end-to-end alignment)
uv run python scripts/install_cupy.py                         # detect GPU/CUDA and install the matching CuPy wheel (dev only; not shipped in the wheel)
```

## Notes for the agent

- The `__main__` self-checks and the `_<func>_cupy_selfcheck()` functions are
  intentionally kept runnable for GPU developers but are excluded from the
  coverage gate (`# pragma: no cover`) — see `CONVENTIONS.md` "GPU coverage
  rule". Do not remove them; do not try to push them under the ≥90% gate.
- `requires-python = ">=3.12"`. The package targets Python 3.12+.
- Public callables are imported from their submodules explicitly
  (`from imageProcToolkit.X import Y`), not from the package root — keep the
  package `__init__` thin (it only carries `__version__` and a docstring).
- **Self-checks are the arbiter.** When changing an estimator or warper, run the
  module's `_<moduleName>_selfcheck()` and confirm `PASS` with the recovered
  transform matching the synthetic ground truth to the documented tolerance.
- `_phaseCorrelationCore` owns the cupy GPU gate (`_HAVE_CUPY_GPU`); the
  estimator modules re-bind its primitives at import. Do not reference `cp` /
  `_HAVE_CUPY_GPU` from the estimator modules — keep the dispatch inside
  `_phaseCorrelationCore`.