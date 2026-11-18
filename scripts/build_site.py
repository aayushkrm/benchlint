"""Build the static GitHub Pages site from committed results (site/dist).

The index page holds the E1, E2 and E3 tables (rendered from results/**/*.json, full mode only).
Each E3 dataset gets its own audit report page, written by benchlint's HTML report writer. The
result JSON is copied alongside so that every number on the site can be traced to a file.
With no full results the site says so; the build still succeeds.

Usage:
    python scripts/build_site.py                      # writes site/dist
    python scripts/build_site.py --out some/dir --results some/results
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchlint.experiments import REPO_ROOT, RESULTS_DIR, render_index_html  # noqa: E402
from benchlint.report import write_reports  # noqa: E402
from benchlint.schema import AuditReport  # noqa: E402
from render_readme import load_results  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "site" / "dist"
TITLE = "benchlint: benchmark audit results"


def build(results_dir: Path, out_dir: Path) -> list[Path]:
    """Write the site into ``out_dir`` and return the paths written."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    res = load_results(results_dir)
    written: list[Path] = []

    links: list[tuple[str, str, str]] = []
    e3 = res["e3"]
    if e3 is not None:
        for entry in e3["datasets"]:
            if entry.get("status") != "ok":
                continue
            report = AuditReport.model_validate_json(
                (results_dir / "e3" / entry["report"]).read_text(encoding="utf-8")
            )
            target = out_dir / "e3" / entry["name"]
            written += write_reports(report, target, formats=("html",))
            links.append((entry["name"], f"e3/{entry['name']}/report.html", "ok"))

    index = out_dir / "index.html"
    index.write_text(
        render_index_html(TITLE, res["calib"], res["e2"], e3, links, res["e1_summary"]),
        encoding="utf-8",
    )
    written.append(index)

    copied = copy_full_results(results_dir, out_dir / "results")
    written += copied
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    written.append(out_dir / ".nojekyll")
    return written


def copy_full_results(results_dir: Path, dest: Path) -> list[Path]:
    """Copy result JSON whose mode is full. Smoke output and stage files are not published."""
    copied: list[Path] = []
    for src in sorted(results_dir.rglob("*.json")):
        data: Any = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("mode") != "full":
            continue
        target = dest / src.relative_to(results_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied.append(target)
    return copied


def main(argv: list[str] | None = None) -> int:
    """Build the site and report what was written."""
    parser = argparse.ArgumentParser(description="Build the GitHub Pages site.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    written = build(args.results, args.out)
    full = any(p.suffix == ".json" for p in written)
    print(f"site: wrote {len(written)} files to {args.out}", file=sys.stderr)
    if not full:
        print("site: no full-mode results yet; the index says so", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
