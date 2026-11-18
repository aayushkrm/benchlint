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
from benchlint.headline import build_headline, headline_text  # noqa: E402

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
    """Write the README blocks and results/headline.json, or with --check compare and exit 1.

    The headline is built from the same results and validated against
    schemas/headline.schema.json before it is written or compared.
    """
    parser = argparse.ArgumentParser(description="Render results into README.md and headline.json.")
    parser.add_argument("--check", action="store_true", help="fail if a rendered file is stale")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--headline", action="store_true", help="write or check results/headline.json only"
    )
    args = parser.parse_args(argv)

    headline_path = args.results / "headline.json"
    headline_new = headline_text(build_headline(args.results))
    current: str | None = None
    readme_new: str | None = None
    if not args.headline:
        current = args.readme.read_text(encoding="utf-8")
        readme_new = splice(current, render_blocks(args.results))
    if args.check:
        stale: list[str] = []
        if readme_new is not None and readme_new != current:
            stale.append(str(args.readme))
        on_disk = headline_path.read_text(encoding="utf-8") if headline_path.is_file() else None
        if on_disk != headline_new:
            stale.append(str(headline_path))
        if stale:
            names = ", ".join(stale)
            print(
                f"out of date with results/: {names}. Run make report and commit.", file=sys.stderr
            )
            return 1
        print("results blocks and headline are up to date", file=sys.stderr)
        return 0
    headline_path.parent.mkdir(parents=True, exist_ok=True)
    headline_path.write_text(headline_new, encoding="utf-8")
    if readme_new is not None:
        args.readme.write_text(readme_new, encoding="utf-8")
    print(f"wrote {headline_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
