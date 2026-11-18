"""E3: leakage scan of six public benchmarks (docs/prereg.md, E3).

Runs the committed specs in specs/ through benchlint's audit (the same run_audit used by
`benchlint audit`). Each spec is one dataset; mbpp-vs-humaneval also checks MBPP against the
HumanEval prompts (13-gram). Settings are the committed specs and are not changed here.

Usage:
    python scripts/run_e3.py            # full run, writes results/e3/
    python scripts/run_e3.py --smoke    # gsm8k and mbpp-vs-humaneval, 200 rows, temp folder
    python scripts/run_e3.py --force    # recompute stored reports

Outputs: results/e3/<spec>.json (AuditReport), results/e3/summary.json and results/e3/env.json.
A dataset that fails to load is recorded in the summary with its error and the run continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import envinfo  # noqa: E402
from benchlint.cli import load_audit_spec, run_audit  # noqa: E402
from benchlint.experiments import (  # noqa: E402
    MODE_FULL,
    MODE_SMOKE,
    REPO_ROOT,
    RESULTS_DIR,
    config_hash,
    write_json,
)
from benchlint.report import render_json, wilson_interval  # noqa: E402
from benchlint.schema import AuditReport  # noqa: E402

SPECS: tuple[str, ...] = (
    "gsm8k",
    "arc-easy",
    "ag-news",
    "imdb",
    "snli",
    "mbpp-vs-humaneval",
)
SMOKE_SPECS: tuple[str, ...] = ("gsm8k", "mbpp-vs-humaneval")
SMOKE_CAP = 200


def _summarise(report: AuditReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in report.findings:
        lo, hi = wilson_interval(f.count, f.denominator)
        rows.append(
            {
                "check": f.check,
                "severity": f.severity,
                "splits": f.splits,
                "count": f.count,
                "denominator": f.denominator,
                "rate": f.rate,
                "ci_low": lo,
                "ci_high": hi,
                "n_examples": len(f.examples),
            }
        )
    return rows


def run_spec(name: str, mode: str) -> tuple[AuditReport, float]:
    """Audit one committed spec. In smoke mode the row cap is replaced by SMOKE_CAP."""
    audit = load_audit_spec(REPO_ROOT / "specs" / f"{name}.yaml")
    if mode == MODE_SMOKE:
        audit = audit.model_copy(
            update={"dataset": audit.dataset.model_copy(update={"max_rows_per_split": SMOKE_CAP})}
        )
    start = time.perf_counter()
    report = run_audit(audit, REPO_ROOT / "specs")
    return report, round(time.perf_counter() - start, 3)


def main(argv: list[str] | None = None) -> int:
    """Run E3 over the committed specs and write the reports and the summary."""
    parser = argparse.ArgumentParser(description="E3 leakage scan.")
    parser.add_argument("--smoke", action="store_true", help="two datasets, tiny caps")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument("--force", action="store_true", help="recompute stored reports")
    args = parser.parse_args(argv)

    mode = MODE_SMOKE if args.smoke else MODE_FULL
    out = args.out
    if out is None:
        out = (
            Path(tempfile.mkdtemp(prefix="benchlint-smoke-e3-"))
            if args.smoke
            else RESULTS_DIR / "e3"
        )
    out.mkdir(parents=True, exist_ok=True)
    names = SMOKE_SPECS if args.smoke else SPECS

    old = {}
    summary_path = out / "summary.json"
    if summary_path.is_file() and not args.force:
        old = {
            d["name"]: d for d in json.loads(summary_path.read_text(encoding="utf-8"))["datasets"]
        }

    datasets: list[dict[str, Any]] = []
    for name in names:
        spec_bytes = (REPO_ROOT / "specs" / f"{name}.yaml").read_bytes()
        chash = config_hash({"spec": spec_bytes.decode("utf-8"), "mode": mode})
        report_path = out / f"{name}.json"
        prev = old.get(name)
        if (
            prev
            and prev.get("config_hash") == chash
            and prev.get("status") == "ok"
            and report_path.is_file()
        ):
            print(f"{name}: reusing stored report", file=sys.stderr)
            datasets.append(prev)
            continue
        entry: dict[str, Any] = {"name": name, "spec": f"specs/{name}.yaml", "config_hash": chash}
        try:
            report, runtime = run_spec(name, mode)
        except Exception as exc:  # recorded, never silently dropped
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name}: ERROR {entry['error']}", file=sys.stderr)
        else:
            report_path.write_text(render_json(report), encoding="utf-8")
            entry.update(
                {
                    "status": "ok",
                    "error": None,
                    "report": report_path.name,
                    "n_records": sum(
                        d["kept_rows"] for d in report.env_info.get("data_splits", [])
                    ),
                    "findings": _summarise(report),
                    "shortcut_flagged": sum(1 for s in report.shortcuts if s.flagged),
                    "runtime_s": runtime,
                }
            )
            print(f"{name}: {len(report.findings)} findings in {runtime} s", file=sys.stderr)
        datasets.append(entry)

    write_json(
        summary_path,
        {"experiment": "E3", "mode": mode, "datasets": datasets},
    )
    write_json(out / "env.json", envinfo.collect())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
