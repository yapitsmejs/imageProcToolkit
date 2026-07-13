## Summary

<!-- What does this PR change, and why? -->

## Test-driven work

- [ ] Tests written/extended first and run red
      (`uv run pytest tests/test_<module>.py`)
- [ ] Minimal implementation added to turn them green, then refactored
- [ ] `uv run python scripts/check.py` passes
      (ruff + format + mypy + tests + ≥90% branch coverage)

## Checklist

- [ ] Public functions have type hints + Google-style docstrings
- [ ] `ruff format` applied; no new lint findings
- [ ] Naming follows `camelCase` + `_aspect` suffix (see `CONVENTIONS.md`)