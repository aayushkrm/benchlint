"""Render committed result JSON into README.md between the results markers.

The README has two marker pairs, in this order: the E1 calibration block first, then the E2
and E3 findings block. Numbers come only from results/**/*.json (full mode). With no full
results the blocks say so.

Usage:
    python scripts/render_readme.py           # write README.md
    python scripts/render_readme.py --check   # exit 1 if README.md is out of date (CI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchlint.experiments import (  # noqa: E402
    REPO_ROOT,
    RESULTS_DIR,
    calibration_block,
    findings_block,
    load_full,
    read_json,
)

START = "<!-- results:start -->"
END = "<!-- results:end -->"
E2_DATASETS = ("snli", "mnli")


def load_results(results_dir: Path) -> dict[str, Any]:
    """Full-mode results and their environment records, or None where absent."""
    e2 = [load_full(results_dir / "e2" / f"{name}.json") for name in E2_DATASETS]
    return {
        "calib": load_full(results_dir / "e1" / "calibration.json"),
        "e1_summary": load_full(results_dir / "e1" / "summary.json"),
        "env_e1": read_json(results_dir / "e1" / "env.json"),
        "e2": [r for r in e2 if r is not None],
        "env_e2": read_json(results_dir / "e2" / "env.json"),
        "e3": load_full(results_dir / "e3" / "summary.json"),
    }


def render_blocks(results_dir: Path) -> list[str]:
    """The two README blocks, in marker order."""
    res = load_results(results_dir)
    return [
        calibration_block(res["calib"], res["env_e1"], res["e1_summary"]),
        findings_block(res["e2"], res["e3"], res["env_e2"]),
    ]


def splice(text: str, blocks: list[str]) -> str:
    """Replace the content of each marker pair, in order, with the matching block.

    Raises ``ValueError`` unless the text has exactly ``len(blocks)`` marker pairs.
    """
    parts: list[str] = []
    pos = 0
    index = 0
    while True:
        start = text.find(START, pos)
        if start < 0:
            break
        if index >= len(blocks):
            raise ValueError("README has more results markers than blocks")
        end = text.find(END, start + len(START))
        if end < 0:
            raise ValueError("a results:start marker has no matching results:end")
        parts.append(text[pos : start + len(START)])
        parts.append("\n" + blocks[index] + "\n")
        pos = end
        index += 1
    if index != len(blocks):
        raise ValueError(f"README has {index} results blocks; expected {len(blocks)}")
    parts.append(text[pos:])
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Write the rendered blocks into the README, or with --check compare and exit 1."""
    parser = argparse.ArgumentParser(description="Render results into README.md.")
    parser.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    current = args.readme.read_text(encoding="utf-8")
    rendered = splice(current, render_blocks(args.results))
    if args.check:
        if rendered != current:
            print(
                f"{args.readme} is out of date with results/: run `make report` and commit.",
                file=sys.stderr,
            )
            return 1
        print("results blocks are up to date", file=sys.stderr)
        return 0
    args.readme.write_text(rendered, encoding="utf-8")
    print(f"wrote results blocks to {args.readme}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
