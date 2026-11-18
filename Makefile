# Run `make help`-style targets from the repo root. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup lint fmt type test demo report check-results site release

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

demo:
	uv run benchlint demo --out demo_out

report:
	@echo "report: scripts/render_readme.py is added in the integration work package" >&2
	@exit 1

check-results:
	@echo "check-results: scripts/render_readme.py --check is added in the integration work package" >&2
	@exit 1

site:
	@echo "site: the static site builder is added in the integration work package" >&2
	@exit 1

release:
	@echo "release: not yet implemented; public release waits for the publish gate" >&2
	@exit 1
