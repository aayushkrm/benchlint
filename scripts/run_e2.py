"""E2: hypothesis-only accuracy on SNLI and MNLI against reported literature values.

Question and settings: docs/prereg.md, E2. Data and caps come from the committed specs
(specs/snli.yaml, specs/mnli.yaml), so the dataset definition is in one place. Label -1 is
dropped before the seeded cap. The hypothesis-only view is benchlint's partial-input baseline
(TF-IDF with 1 and 2-grams, then logistic regression).

Usage:
    python scripts/run_e2.py            # full run, writes results/e2/
    python scripts/run_e2.py --smoke    # tiny sizes, under 60 s, writes to a temp folder
    python scripts/run_e2.py --force    # recompute even if a stored result matches

Outputs: results/e2/snli.json, results/e2/mnli.json, results/e2/env.json. A dataset whose
settings are unchanged is skipped on re-run (resume).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import envinfo  # noqa: E402
from benchlint.checks.shortcut import run_shortcut_checks  # noqa: E402
from benchlint.cli import load_audit_spec  # noqa: E402
from benchlint.experiments import (  # noqa: E402
    LITERATURE_CITATION,
    MODE_FULL,
    MODE_SMOKE,
    REPO_ROOT,
    RESULTS_DIR,
    config_hash,
    read_if_same_config,
    write_json,
)
from benchlint.io import load_dataset_detailed  # noqa: E402

VIEW = "hypothesis_only"
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "snli": {
        "spec": "specs/snli.yaml",
        "train": "train",
        "test": "test",
        "label": "SNLI hypothesis-only accuracy",
        "reported": 0.67,
    },
    "mnli": {
        "spec": "specs/mnli.yaml",
        "train": "train",
        "test": "validation",
        "label": "MNLI matched dev hypothesis-only accuracy",
        "reported": 0.53,
    },
}
SMOKE_CAP = 300
SMOKE_BOOT = 200
SMOKE_PERM = 20


def run_one(name: str, mode: str, chash_extra: dict[str, Any]) -> dict[str, Any]:
    """Run the hypothesis-only baseline on one dataset and return its result object."""
    cfg = EXPERIMENTS[name]
    smoke = mode == MODE_SMOKE
    audit = load_audit_spec(REPO_ROOT / cfg["spec"])
    ds = audit.dataset
    shortcut = audit.checks.shortcut
    if smoke:
        ds = ds.model_copy(update={"max_rows_per_split": SMOKE_CAP})
        shortcut = shortcut.model_copy(
            update={"max_rows": SMOKE_CAP, "n_boot": SMOKE_BOOT, "n_perm": SMOKE_PERM}
        )
    start = time.perf_counter()
    splits, loads = load_dataset_detailed(ds, base_dir=REPO_ROOT / "specs")
    records = [r for rs in splits.values() for r in rs]
    results = run_shortcut_checks(
        records,
        shortcut,
        train_split=cfg["train"],
        test_split=cfg["test"],
        seed=ds.seed,
    )
    hyp = next(r for r in results if r.view == VIEW)
    load_by_split = {x.split: x for x in loads}
    gap = (hyp.value - cfg["reported"]) * 100
    return {
        "experiment": "E2",
        "dataset": name,
        "mode": mode,
        "config_hash": config_hash({"dataset": name, **chash_extra}),
        "source": ds.source.id_or_path,
        "config": ds.source.config,
        "spec": cfg["spec"],
        "view": VIEW,
        "train_split": cfg["train"],
        "eval_split": cfg["test"],
        "n_train": hyp.n_train,
        "n_eval": hyp.n_test,
        "train_source_rows": load_by_split[cfg["train"]].source_rows,
        "train_dropped_by_label": load_by_split[cfg["train"]].dropped_by_label,
        "eval_dropped_by_label": load_by_split[cfg["test"]].dropped_by_label,
        "accuracy": {"value": hyp.value, "ci_low": hyp.ci_low, "ci_high": hyp.ci_high},
        "majority": hyp.majority,
        "chance": hyp.chance,
        "p_value": hyp.p_value,
        "flagged": hyp.flagged,
        "literature": {
            "value": cfg["reported"],
            "metric": "accuracy",
            "status": "reported",
            "setting": cfg["label"],
            "citation": LITERATURE_CITATION,
        },
        "gap_points": round(gap, 4),
        "shortcuts": [r.model_dump(mode="json") for r in results],
        "runtime_s": round(time.perf_counter() - start, 3),
        "smoke_caps": {"max_rows": SMOKE_CAP} if smoke else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run E2 for SNLI and MNLI and write one JSON per dataset plus env.json."""
    parser = argparse.ArgumentParser(description="E2 hypothesis-only replication.")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, temp output folder")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument("--force", action="store_true", help="recompute stored results")
    args = parser.parse_args(argv)

    mode = MODE_SMOKE if args.smoke else MODE_FULL
    out = args.out
    if out is None:
        out = (
            Path(tempfile.mkdtemp(prefix="benchlint-smoke-e2-"))
            if args.smoke
            else RESULTS_DIR / "e2"
        )
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in EXPERIMENTS.items():
        spec_text = (REPO_ROOT / cfg["spec"]).read_text(encoding="utf-8")
        extra = {"spec_text": spec_text, "reported": cfg["reported"], "mode": mode}
        chash = config_hash({"dataset": name, **extra})
        path = out / f"{name}.json"
        if not args.force and read_if_same_config(path, chash) is not None:
            print(f"{name}: reusing stored result", file=sys.stderr)
            continue
        result = run_one(name, mode, extra)
        result["config_hash"] = chash
        write_json(path, result)
        print(
            f"{name}: accuracy {result['accuracy']['value']:.4f} on {result['n_eval']} eval items",
            file=sys.stderr,
        )
    write_json(out / "env.json", envinfo.collect())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
