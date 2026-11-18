# Run `make help`-style targets from the repo root. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup lint fmt type test demo results summary report check-results site release

setup:
	uv venv -p 3.11 .venv
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

type:
	uv run mypy src

test:
	uv run pytest -m "not model"

# Synthetic demo with planted leaks. Exit 1 is the expected result for findings, so the demo
# runs with --fail-on none and exits 0 when it completes.
demo:
	uv run benchlint demo --out demo_out --fail-on none

# The full experiment queue, in order. Slow (network downloads, CPU-bound); not run in CI.
results:
	uv run python scripts/calibrate.py
	uv run python scripts/run_e2.py
	uv run python scripts/run_e3.py

# E1 summary (LSH recall, true-Jaccard bins, length arithmetic) from the stored recall stage.
summary:
	uv run python scripts/summarise_e1.py

# Render committed result JSON into README.md (only the results markers change).
report:
	uv run python scripts/render_readme.py

# CI gate: fails if README.md is out of date with results/.
check-results:
	uv run python scripts/render_readme.py --check

# Static GitHub Pages site from committed results (index and per-dataset reports).
site:
	uv run python scripts/build_site.py

# Build the source and wheel distributions into dist/.
release:
	uv build
