# AGENTS.md

Guide for coding agents and contributors working on benchlint.

## What this repo is

benchlint is a linter for ML benchmarks. It finds train/test leakage, exact and near
duplicates, label conflicts, n-gram contamination and shortcut features before a benchmark
score is trusted. Python 3.11+, src layout, MIT licence.

## Setup and commands

```
make setup        # uv venv -p 3.11 .venv && uv sync --extra dev
make lint         # ruff check + ruff format --check
make fmt          # ruff fix + format
make type         # mypy src
make test         # pytest -m "not model"
make demo         # benchlint demo (synthetic data, under 30 s, no downloads)
```

CI (`.github/workflows/ci.yml`) runs ruff, mypy, pytest and `make check-results` on Linux for Python 3.11 and 3.12.

## Layout

```
src/benchlint/
  schema.py        pydantic models for specs and reports (see invariant 1)
  io.py            loaders: HF (with parquet fallback), JSONL, CSV, parquet
  normalize.py     text normalisation
  checks/          exact, near, ngram, labels, shortcut, canary
  stats.py         seeded bootstrap, permutation p-values, ROC-AUC
  report.py        JSON, Markdown, HTML writers; --fail-on logic
  cli.py           typer app: audit, check, canary, demo
  demo_data/       synthetic dataset with planted leaks (labelled SYNTHETIC)
specs/             DatasetSpec YAML files, one per dataset
tests/             pytest; tests/test_<module>.py next to the module it tests
scripts/envinfo.py environment provenance JSON (no hostname, no username)
results/           committed result JSON (generated, never hand-edited)
docs/              design, prereg (committed before runs), deviations, limitations
.claude/skills/benchlint/SKILL.md   how to use the tool (for Claude Code)
```

## Invariants

1. Change `src/benchlint/schema.py` only with a `schema_version` decision and an entry in
   `docs/deviations.md`. Until then, keep a local adapter rather than editing the models.
2. `results/` is generated. Never type a measured number into README, docs or the site.
   Numbers come from result JSON through the render script.
3. Every result carries `env.json` or equivalent provenance from `scripts/envinfo.py`.
4. Tests run without network access. Tests needing a local model or a download carry
   `@pytest.mark.model` (or a download marker) and are excluded in CI.
5. Pre-register experiments in `docs/prereg.md` before running them. Log every change of n,
   threshold, prompt or model in `docs/deviations.md`.
6. Uncertainty comes with every rate: n, and a 95% CI (Wilson or seeded bootstrap).
7. Synthetic output (demo data) is labelled SYNTHETIC.
8. Exit codes: 0 pass, 1 check failed (finding at or above `--fail-on`), 2 usage or runtime
   error.
9. Do not use the words "production-ready", "state-of-the-art", "novel", "robust",
   "enterprise" or "first" in docs unless a measured result supports them.

## Style

- ruff: line length 100; rules E, F, I, UP, B.
- mypy: `disallow_untyped_defs`; `src` only.
- Public functions have type hints and a docstring.
- Seeds are explicit parameters; no global random state.
