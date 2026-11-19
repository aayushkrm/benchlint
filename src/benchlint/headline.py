"""Headline numbers for the portfolio hub: results/headline.json (hub standard contract).

Every value is formatted from full-mode result files. Nothing is typed by hand. The file has at
most six items: two from E1, two from E2, one from E3 and one from E4 (E3 shows its top
finding; the second-ranked one is in the README tables). Items whose result file is missing are
left out, so an empty results folder gives an empty item list that still validates.

Selection rule for E3 (deterministic): for each benchmark, the cross-split exact or near finding
with the highest count (ties: higher rate, then exact before near). Across benchmarks, the two
largest count (ties: higher rate, then benchmark name). The top item is featured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchlint.experiments import (
    REPO_ROOT,
    load_full,
    pct,
    read_json,
)
from benchlint.jsonsubset import validate
from benchlint.report import wilson_interval

SCHEMA_VERSION = "1"
TOOL = "benchlint"
MAX_ITEMS = 6
E3_ITEMS = 1
E4_ITEMS = 1
TAGLINE = (
    "A linter for ML benchmarks: finds train/test leakage, near-duplicates, label conflicts "
    "and shortcut features before you trust a score."
)
QUICKSTART = "uv tool install . && benchlint demo"
LINKS: dict[str, Any] = {
    "repo": "https://github.com/aayushkrm/benchlint",
    "site": "https://aayushkrm.github.io/benchlint/",
    "release": None,
}
SCHEMA_PATH = REPO_ROOT / "schemas" / "headline.schema.json"
E2_NAMES: dict[str, str] = {"snli": "SNLI test", "mnli": "MNLI matched dev"}
CROSS_SCOPES = ("cross", "cross_split")


def load_schema() -> dict[str, Any]:
    """The committed headline schema."""
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{SCHEMA_PATH} is not a JSON object")
    return data


def _row(rows: list[dict[str, Any]], **want: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in want.items()):
            return row
    return None


def environment_text(env: dict[str, Any] | None) -> str:
    """One line describing the machine of the E1 run, from results/e1/env.json."""
    if env is None:
        return "No environment record yet (results/e1/env.json is missing)."
    parts: list[str] = [str(env.get("cpu_model") or "unknown CPU")]
    phys, logical = env.get("cores_physical"), env.get("cores_logical")
    if phys and logical:
        parts.append(f"{phys} physical / {logical} logical cores")
    ram = env.get("ram_bytes")
    if ram:
        parts.append(f"{round(ram / 2**30)} GB RAM")
    os_name, os_version = env.get("os_name"), env.get("os_version")
    if os_name == "Darwin" and os_version:
        os_label = f"macOS {os_version}"
    else:
        os_label = " ".join(str(x) for x in (os_name, os_version) if x)
    machine = env.get("machine")
    parts.append(" ".join(str(x) for x in (os_label, machine) if x))
    parts.append("CPU only")
    if env.get("python_version"):
        parts.append(f"Python {env['python_version']}")
    return ", ".join(parts) + " (environment of the E1 run, results/e1/env.json)"


def _e1_items(results_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    summary = load_full(results_dir / "e1" / "summary.json")
    calib = load_full(results_dir / "e1" / "calibration.json")
    if summary is not None:
        lsh = summary["lsh_recall_at_threshold"]
        overall = summary["all_near"]
        threshold = summary["threshold"]
        items.append(
            {
                "id": "e1_lsh_recall",
                "label": (
                    f"E1 detector recall for pairs at or above Jaccard {threshold} (LSH recall)"
                ),
                "value": (
                    f"{pct(lsh['recall'])} at true Jaccard at or above {threshold} "
                    f"({lsh['found']} of {lsh['n']}); overall {pct(overall['recall'])} "
                    f"({overall['found']} of {overall['n']})"
                ),
                "detail": (
                    "Planted near-duplicates (synthetic word edits of AG News test items) whose "
                    f"exact {summary['shingle']}-word shingle Jaccard is at or above {threshold}. "
                    "Only these pairs can be reported. Missed by LSH retrieval: "
                    f"{summary['missed_at_or_above_threshold']}. Reported below the threshold: "
                    f"{summary['found_below_threshold']}."
                ),
                "n": lsh["n"],
                "ci": f"[{pct(lsh['ci_low'])}, {pct(lsh['ci_high'])}]",
                "synthetic": True,
                "featured": True,
                "source": "results/e1/summary.json",
            }
        )
    if calib is not None:
        row = _row(calib["rows"], edit="all", size="all", length="short")
        if row is not None and row["n"] > 0:
            bin_label = calib["settings"]["length_bins"]["short"]
            items.append(
                {
                    "id": "e1_short_texts",
                    "label": f"E1 detector recall, short texts ({bin_label}), all edits",
                    "value": f"{pct(row['recall'])} ({row['found']} of {row['n']})",
                    "detail": (
                        "Share of all planted near-duplicates in the shortest length bin that "
                        "were found, across all edit types and sizes. Low by arithmetic: one "
                        "word edit changes up to five 5-gram shingles, so short texts fall below "
                        "the reporting threshold (README, docs/limitations.md)."
                    ),
                    "n": row["n"],
                    "ci": f"[{pct(row['ci_low'])}, {pct(row['ci_high'])}]",
                    "synthetic": True,
                    "featured": True,
                    "source": "results/e1/calibration.json",
                }
            )
    return items


def _e2_items(results_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, label in E2_NAMES.items():
        r = load_full(results_dir / "e2" / f"{name}.json")
        if r is None:
            continue
        acc = r["accuracy"]
        lit = r["literature"]
        items.append(
            {
                "id": f"e2_{name}",
                "label": f"E2 {label} hypothesis-only accuracy vs literature",
                "value": (
                    f"{pct(acc['value'])} accuracy (literature {pct(lit['value'], 1)}, reported)"
                ),
                "detail": (
                    f"Hypothesis-only TF-IDF and logistic regression, train capped at "
                    f"{r['n_train']} rows, majority baseline {pct(r['majority'])}. Gap "
                    f"{r['gap_points']:+.1f} points against the literature value, which is cited "
                    "as reported (Gururangan et al. 2018, NAACL, Table 2) and not measured here."
                ),
                "n": r["n_eval"],
                "ci": f"[{pct(acc['ci_low'])}, {pct(acc['ci_high'])}]",
                "synthetic": False,
                "featured": True,
                "source": f"results/e2/{name}.json",
            }
        )
    return items


def select_cross_findings(
    per_dataset: list[tuple[str, list[dict[str, Any]]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Choose the E3 findings to show, as (benchmark, finding) pairs, in rank order.

    ``per_dataset`` holds each benchmark's findings (AuditReport dicts). Only exact and near
    findings whose scope is cross-split are considered. The rule is in the module docstring.
    """
    best: list[tuple[str, dict[str, Any]]] = []
    for name, findings in per_dataset:
        cands = [
            f
            for f in findings
            if f["check"] in ("exact", "near") and f.get("details", {}).get("scope") in CROSS_SCOPES
        ]
        if not cands:
            continue
        chosen = sorted(cands, key=lambda f: (-f["count"], -f["rate"], f["check"]))[0]
        best.append((name, chosen))
    ranked = sorted(best, key=lambda c: (-c[1]["count"], -c[1]["rate"], c[0]))
    return ranked[:E3_ITEMS]


def _e3_items(results_dir: Path) -> list[dict[str, Any]]:
    summary = load_full(results_dir / "e3" / "summary.json")
    if summary is None:
        return []
    per_dataset: list[tuple[str, list[dict[str, Any]]]] = []
    for entry in summary["datasets"]:
        if entry.get("status") != "ok":
            continue
        report = read_json(results_dir / "e3" / entry["report"])
        if report is None:
            continue
        per_dataset.append((entry["name"], report["findings"]))
    items: list[dict[str, Any]] = []
    for rank, (name, f) in enumerate(select_cross_findings(per_dataset)):
        kind = "exact" if f["check"] == "exact" else "near"
        lo, hi = wilson_interval(f["count"], f["denominator"])
        items.append(
            {
                "id": f"e3_{name}_cross_{kind}",
                "label": f"E3 {name}: cross-split {kind} duplicates",
                "value": f"{f['count']} of {f['denominator']} items ({pct(f['rate'], 2)})",
                "detail": (
                    "Largest cross-split duplicate count among the six E3 benchmarks. Selection: "
                    "per benchmark, the cross-split exact or near finding with the highest count; "
                    "then the largest count across benchmarks. Exact counts test items that "
                    "repeat a train item (denominator: the measured split). Near counts items with "
                    "a 5-gram Jaccard partner at or above 0.8 in the other split (denominator: all "
                    "evaluable items of both splits). Matching is lexical; shared templates can "
                    "match without contamination."
                ),
                "n": f["denominator"],
                "ci": f"[{pct(lo, 2)}, {pct(hi, 2)}]",
                "synthetic": False,
                "featured": rank == 0,
                "source": f"results/e3/{name}.json",
            }
        )
    return items


def _e4_items(results_dir: Path) -> list[dict[str, Any]]:
    """E4 dogfooding: quantgate's ARC-Easy eval subset against its train split (pre-registered)."""
    summary = load_full(results_dir / "e4" / "summary.json")
    if summary is None:
        return []
    arc = summary["datasets"]["quantgate_arc_easy"]
    near = arc["checks"]["near"]
    exact = arc["checks"]["exact"]
    n = arc["n_eval"]
    lo, hi = wilson_interval(near["matched"], n)
    supp = arc["supplementary"]["checks"]["near"]["matched"]
    passk = summary["datasets"]["passk"]
    pw = passk["pairwise"]
    return [
        {
            "id": "e4_quantgate_arc_easy_near",
            "label": "E4 dogfooding: quantgate ARC-Easy eval items with a train near-duplicate",
            "value": f"{near['matched']} of {n} ({pct(near['rate'], 2)}); pre-registered text",
            "detail": (
                f"The committed {n}-item quantgate eval subset against the ARC-Easy train split, "
                "on "
                "the text quantgate's prompt holds (question and choices), at the E3 thresholds. "
                f"Exact matches: {exact['matched']}. Supplementary, not pre-registered (E3's "
                f"question-only text): {supp} of {n}. Passk: "
                f"{passk['near']['matched']} of {passk['n_tasks']} instructions have a near "
                f"duplicate; highest pairwise 5-gram Jaccard {pw['max']:.2f}. A null result for "
                "near duplicates here. Small n: these are observations about these inputs, not a "
                "benchmark result."
            ),
            "n": n,
            "ci": f"[{pct(lo, 2)}, {pct(hi, 2)}]",
            "synthetic": False,
            "featured": False,
            "source": "results/e4/quantgate_arc_easy.json",
        }
    ]


def build_headline(results_dir: Path) -> dict[str, Any]:
    """The headline object for ``results_dir``, validated against the committed schema."""
    items = [
        *_e1_items(results_dir),
        *_e2_items(results_dir),
        *_e3_items(results_dir),
        *_e4_items(results_dir),
    ]
    if len(items) > MAX_ITEMS:
        raise ValueError(f"headline has {len(items)} items; the contract allows {MAX_ITEMS}")
    headline: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "tagline": TAGLINE,
        "quickstart": QUICKSTART,
        "environment": environment_text(read_json(results_dir / "e1" / "env.json")),
        "items": items,
        "links": dict(LINKS),
    }
    errors = validate(headline, load_schema())
    if errors:
        raise ValueError(
            "headline does not match schemas/headline.schema.json: " + "; ".join(errors)
        )
    return headline


def headline_text(headline: dict[str, Any]) -> str:
    """The file contents: sorted keys, two-space indent, final newline (as write_json)."""
    return json.dumps(headline, indent=2, sort_keys=True, default=str) + "\n"
