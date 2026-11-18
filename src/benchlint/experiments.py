"""Shared helpers for the experiment scripts and for the README and site renderers.

Experiment results are JSON files under ``results/<exp>/``. Each experiment writes a
``config_hash`` (so an interrupted run resumes only when its settings are unchanged) and a
``mode`` of ``"full"`` or ``"smoke"``. Only ``"full"`` results are rendered: smoke output is a
code check with tiny sizes and must never appear in a README or on the site.

Numbers appear in the rendered text only as values read from these JSON files. The text
around them is fixed wording.
"""

# ruff: noqa: E501  (the inline HTML template and the long wording constants are kept whole)

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchlint.report import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
""" Default folder of committed results (the scripts take an --out override for smoke runs)."""
MODE_FULL = "full"
MODE_SMOKE = "smoke"
LOW_N = 20
"""Cells with fewer planted pairs than this are marked "low n" in the E1 tables."""
NO_RESULTS = "_No results yet. The experiments have not been run in full: `make results`, then `make report`._"
SYNTHETIC_EDITS_NOTE = (
    "SYNTHETIC: the planted copies are generated edits of real AG News text. They measure the "
    "detector on known edits, not on real leaks."
)
LITERATURE_CITATION = (
    "Gururangan et al. (2018), Annotation Artifacts in Natural Language Inference Data, NAACL, "
    "Table 2 (fastText hypothesis-only), arXiv:1803.02324"
)


def config_hash(obj: Any) -> str:
    """Short, stable hash of a JSON-serialisable settings object."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (temporary file, then rename) with sorted keys and a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp_name, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Parsed JSON object at ``path``, or None if the file is missing."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def read_if_same_config(path: Path, chash: str) -> dict[str, Any] | None:
    """The stored result if its ``config_hash`` matches (resume), otherwise None."""
    data = read_json(path)
    if data is None or data.get("config_hash") != chash:
        return None
    return data


def load_full(path: Path) -> dict[str, Any] | None:
    """A result for rendering: None if missing, or if it is not a full-mode result."""
    data = read_json(path)
    if data is None or data.get("mode") != MODE_FULL:
        return None
    return data


def pct(x: float, digits: int = 1) -> str:
    """A proportion as a percentage string."""
    return f"{100 * x:.{digits}f}%"


def count_rate(count: int, n: int) -> str:
    """``count/n (rate, 95% Wilson CI)`` with the denominator always shown."""
    lo, hi = wilson_interval(count, n)
    rate = count / n if n else 0.0
    return f"{count}/{n} ({pct(rate)}, 95% CI {pct(lo)}–{pct(hi)})"


def short_sha(env: dict[str, Any] | None) -> str:
    """First 10 characters of the git SHA recorded in an ``env.json``, or "n/a"."""
    sha = (env or {}).get("git_sha")
    return str(sha)[:10] if sha else "n/a"


@dataclass(frozen=True)
class Table:
    """One table of a rendered block: a title, optional note text, headers and rows."""

    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str = ""


def _find_row(rows: list[dict[str, Any]], **want: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in want.items()):
            return row
    return None


def _edit_label(edit: str) -> str:
    return {"substitution": "substitution", "deletion": "deletion", "insertion": "insertion"}.get(
        edit, edit
    )


def write_e1_summary(e1_dir: Path) -> dict[str, Any]:
    """Compute ``e1_dir/summary.json`` from the stored recall stage. No detector is run.

    Reads ``recall.json`` (planted pairs) and ``calibration.json`` (threshold and shingle size),
    and refuses to combine stage files whose config hashes differ.
    """
    recall = read_json(e1_dir / "recall.json")
    calib = read_json(e1_dir / "calibration.json")
    if recall is None or calib is None:
        raise FileNotFoundError(f"{e1_dir} needs recall.json and calibration.json")
    if recall.get("config_hash") != calib.get("config_hash"):
        raise ValueError("recall.json and calibration.json come from different configurations")
    settings = calib["settings"]
    summary = e1_summary(recall, float(settings["threshold"]), int(settings["shingle"]))
    write_json(e1_dir / "summary.json", summary)
    return summary


JACCARD_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
)
"""Bins of the true (exact) Jaccard of planted near pairs, for the E1 summary. The last bin
includes 1.0. The bin edges do not depend on the threshold."""
SHINGLE_SIZE_FOR_BOUND = 5
BOUND_EDITS: tuple[int, ...] = (1, 3, 5, 10)
BOUND_LENGTHS: tuple[int, ...] = (35, 50, 100)
EDIT_BOUND_SHINGLES = 5
"""Most shingles that one word edit can change. A word touches at most five 5-gram windows."""


def _jaccard_bin_label(index: int) -> str:
    lo, hi = JACCARD_BINS[index]
    close = "]" if index == len(JACCARD_BINS) - 1 else ")"
    return f"[{lo:.1f}, {hi:.1f}{close}"


def _jaccard_bin_index(value: float) -> int:
    for i, (lo, hi) in enumerate(JACCARD_BINS):
        last = i == len(JACCARD_BINS) - 1
        if lo <= value < hi or (last and value == hi):
            return i
    return len(JACCARD_BINS) - 1


def jaccard_lower_bound(n_words: int, edits: int, shingle: int = SHINGLE_SIZE_FOR_BOUND) -> float:
    """Lower bound on the Jaccard of a text and a copy with ``edits`` word edits.

    With ``S = n_words - shingle + 1`` shingles, the copy keeps at least ``S - 5k`` shared
    shingles and has at most ``S + 5k`` in the union, so Jaccard is at least
    ``(S - 5k) / (S + 5k)``. This is arithmetic, not a measurement, and it is clamped at 0.
    """
    shingles = n_words - shingle + 1
    if shingles <= 0:
        return 0.0
    destroyed = EDIT_BOUND_SHINGLES * edits
    return max(0.0, (shingles - destroyed) / (shingles + destroyed))


def guaranteed_length(edits: int, threshold: float, shingle: int = SHINGLE_SIZE_FOR_BOUND) -> int:
    """Fewest words for which the bound above already reaches ``threshold`` for ``edits`` edits.

    Solves ``(S - 5k) / (S + 5k) >= t`` for ``S``: ``S >= 5k (1 + t) / (1 - t)``. For a threshold
    of 0.8 this is ``S >= 45k``, so ``n_words >= 45k + shingle - 1``.
    """
    needed_shingles = EDIT_BOUND_SHINGLES * edits * (1 + threshold) / (1 - threshold)
    return math.ceil(round(needed_shingles, 9)) + shingle - 1


def _near_planted(recall: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in recall["planted_pairs"]:
        orig_id, copy_id, role, edit, size, length, jac, found = row
        if role != "near":
            continue
        out.append(
            {
                "edit": edit,
                "size": size,
                "length": length,
                "jaccard": float(jac),
                "found": bool(found),
            }
        )
    return out


def _matches(point: dict[str, Any], edit: str, size: Any) -> bool:
    edit_ok = edit == "all" or point["edit"] == edit
    size_ok = size == "all" or point["size"] == size
    return bool(edit_ok and size_ok)


def _proportion(found: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(found, n)
    return {
        "n": n,
        "found": found,
        "recall": found / n if n else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "low_n": n < LOW_N,
    }


def e1_summary(
    recall: dict[str, Any], threshold: float, shingle: int = SHINGLE_SIZE_FOR_BOUND
) -> dict[str, Any]:
    """Summary of the E1 recall stage, computed from its planted pairs (no new detector run).

    The conditional recall is the share of planted near pairs found, among those whose TRUE
    exact Jaccard is at or above ``threshold``. Those pairs are the only ones the detector
    can report, so this is the LSH recall at the threshold. Pairs below the threshold are
    counted separately; the detector cannot report them, so they are never a retrieval miss.
    The true Jaccard comes from the stored exact value (shingle sets), not from the detector.
    """
    near = _near_planted(recall)
    above = [p for p in near if p["jaccard"] >= threshold]
    below = [p for p in near if p["jaccard"] < threshold]
    found_below = sum(1 for p in below if p["found"])
    missed_above = sum(1 for p in above if not p["found"])

    by_bin: list[dict[str, Any]] = []
    for i in range(len(JACCARD_BINS)):
        members = [p for p in near if _jaccard_bin_index(p["jaccard"]) == i]
        by_bin.append(
            {
                "bin": _jaccard_bin_label(i),
                "low": JACCARD_BINS[i][0],
                "high": JACCARD_BINS[i][1],
                **_proportion(sum(1 for p in members if p["found"]), len(members)),
            }
        )

    by_edit_size: list[dict[str, Any]] = []
    for edit in ("all", "substitution", "deletion", "insertion"):
        for size in ("all", 1, 3, 5, 10):
            members = [p for p in above if _matches(p, edit, size)]
            all_members = [p for p in near if _matches(p, edit, size)]
            by_edit_size.append(
                {
                    "edit": edit,
                    "size": size,
                    "planted_all": len(all_members),
                    "found_all": sum(1 for p in all_members if p["found"]),
                    "lsh_recall": _proportion(sum(1 for p in members if p["found"]), len(members)),
                }
            )

    rows: list[dict[str, Any]] = []
    for edits in BOUND_EDITS:
        lengths = {str(n): round(jaccard_lower_bound(n, edits, shingle), 4) for n in BOUND_LENGTHS}
        rows.append(
            {
                "edits": edits,
                "guaranteed_length_words": guaranteed_length(edits, threshold, shingle),
                "lower_bound_by_length": lengths,
            }
        )

    return {
        "experiment": "E1",
        "record": "summary",
        "mode": recall["mode"],
        "config_hash": recall["config_hash"],
        "threshold": threshold,
        "shingle": shingle,
        "near_planted": len(near),
        "all_near": _proportion(sum(1 for p in near if p["found"]), len(near)),
        "lsh_recall_at_threshold": _proportion(sum(1 for p in above if p["found"]), len(above)),
        "below_threshold": _proportion(found_below, len(below)),
        "found_below_threshold": found_below,
        "missed_at_or_above_threshold": missed_above,
        "by_true_jaccard": by_bin,
        "by_edit_size": by_edit_size,
        "bound": {
            "shingle": shingle,
            "rule": "lower bound (S - 5k) / (S + 5k), S = words - shingle + 1; arithmetic only",
            "lengths": list(BOUND_LENGTHS),
            "rows": rows,
        },
    }


def calibration_tables(
    calib: dict[str, Any] | None, summary: dict[str, Any] | None = None
) -> list[Table]:
    """The E1 tables: recall by edit and size, the exact copies, FPR, and recall by length."""
    if calib is None:
        return []
    rows: list[dict[str, Any]] = calib["rows"]
    roles = calib["roles"]
    src = calib["source"]
    threshold = calib["settings"]["threshold"]
    recall_rows: list[tuple[str, ...]] = []
    for edit in ("substitution", "deletion", "insertion"):
        for size in (1, 3, 5, 10):
            r = _find_row(rows, edit=edit, size=size, length="all")
            if r is None:
                continue
            recall_rows.append(
                (
                    _edit_label(edit),
                    str(size),
                    f"{r['found']}/{r['n']}",
                    f"{pct(r['recall'])} [{pct(r['ci_low'])}, {pct(r['ci_high'])}]",
                    f"{r['median_jaccard']:.3f}",
                )
            )
    exact = calib["exact"]
    recall_rows.append(
        (
            "exact copy",
            "0",
            f"{exact['found']}/{exact['n']}",
            f"{pct(exact['recall'])} [{pct(exact['ci_low'])}, {pct(exact['ci_high'])}]",
            "1.000",
        )
    )
    fpr = calib["fpr"]
    length_rows: list[tuple[str, ...]] = []
    for edit in ("substitution", "deletion", "insertion"):
        for size in (1, 3, 5, 10):
            cells = [_edit_label(edit), str(size)]
            for length in ("short", "medium", "long"):
                r = _find_row(rows, edit=edit, size=size, length=length)
                if r is None or r["n"] == 0:
                    cells.append("n/a")
                    continue
                mark = " (low n)" if r["low_n"] else ""
                cells.append(f"{r['found']}/{r['n']} = {pct(r['recall'])}{mark}")
            length_rows.append(tuple(cells))
    bins = calib["length_bins"]
    labels = calib["settings"]["length_bins"]
    base_tables: list[Table] = [
        Table(
            title="E1 recall by edit type and size",
            headers=(
                "Edit",
                "Size (words)",
                "Found / planted",
                "Recall [95% CI]",
                "Median exact Jaccard",
            ),
            rows=tuple(recall_rows),
            note=(
                "Exact copies are found by construction: identical text gives an identical key, "
                "so the exact row checks the planting and normalisation, not the detector. "
                f"Base: {src['n_base']} clean items from `{src['dataset']}` split "
                f"`{src['split']}`. Planted: {roles['exact']} exact copies, {roles['near']} "
                f"near-duplicate copies ({roles['per_bucket']} per edit bucket), {roles['clean']} "
                f"clean controls. Reporting threshold {threshold} on exact Jaccard, 5-word "
                "shingles."
            ),
        ),
        Table(
            title="E1 clean false positives",
            headers=("Clean pairs", "Flagged", "Rate [95% CI]"),
            rows=(
                (
                    str(fpr["pairs"]),
                    str(fpr["flagged"]),
                    f"{pct(fpr['rate'], 2)} [{pct(fpr['ci_low'], 2)}, {pct(fpr['ci_high'], 2)}]",
                ),
            ),
            note=(
                f"Random pairs of clean controls, flagged at threshold {threshold}. Random pairs "
                "are almost never near-duplicates, so this is a weak test of false positives. No "
                "hard-negative rate (same-topic or templated pairs) has been measured."
            ),
        ),
        Table(
            title="E1 recall by text length (found / planted)",
            headers=(
                "Edit",
                "Size",
                f"short, {labels['short']} (base n {bins['short']})",
                f"medium, {labels['medium']} (base n {bins['medium']})",
                f"long, {labels['long']} (base n {bins['long']})",
            ),
            rows=tuple(length_rows),
            note=(
                f"Cells with fewer than {LOW_N} planted pairs are marked low n. Bins are word-count "
                "tertiles of the eligible pool. Paraphrase "
                "detection is out of scope: the tool has no language or embedding model."
            ),
        ),
    ]
    if summary is None:
        return base_tables
    return [base_tables[0], *_summary_tables(summary), *base_tables[1:]]


def _summary_tables(summary: dict[str, Any]) -> list[Table]:
    """E1 tables computed by the summary step: LSH recall, recall by true Jaccard, bound."""
    t = summary["threshold"]
    conditional_rows: list[tuple[str, ...]] = []
    for row in summary["by_edit_size"]:
        ci = row["lsh_recall"]
        conditional_rows.append(
            (
                _edit_label(row["edit"]) if row["edit"] != "all" else "all edits",
                "all" if row["size"] == "all" else str(row["size"]),
                str(row["planted_all"]),
                str(ci["n"]),
                f"{ci['found']}/{ci['n']}",
                (
                    "n/a: no planted pairs at or above the threshold"
                    if ci["n"] == 0
                    else f"{pct(ci['recall'])} [{pct(ci['ci_low'])}, {pct(ci['ci_high'])}]"
                    + (" (low n)" if ci["low_n"] else "")
                ),
            )
        )
    conditional = Table(
        title=f"E1 detector recall conditional on true Jaccard >= {t} (LSH recall)",
        headers=(
            "Edit",
            "Size (words)",
            "Planted near pairs",
            "With true Jaccard >= threshold",
            "Found / those pairs",
            "LSH recall [95% CI]",
        ),
        rows=tuple(conditional_rows),
        note=(
            f"Only pairs whose exact Jaccard is at or above {t} can be reported, so this is the "
            "recall of the retrieval step among pairs the detector is allowed to report. Pairs "
            f"below {t} are not counted here; the detector cannot report them by design. "
            f"Above-threshold pairs missed by retrieval: {summary['missed_at_or_above_threshold']}. "
            f"Below-threshold pairs reported: {summary['found_below_threshold']} (must be 0)."
        ),
    )
    bins = Table(
        title="E1 recall by true Jaccard bin",
        headers=("True Jaccard", "Planted near pairs", "Found", "Recall [95% CI]"),
        rows=tuple(
            (
                b["bin"],
                str(b["n"]),
                str(b["found"]),
                f"{pct(b['recall'])} [{pct(b['ci_low'])}, {pct(b['ci_high'])}]"
                + (" (low n)" if b["low_n"] else ""),
            )
            for b in summary["by_true_jaccard"]
        ),
        note=(
            "True Jaccard is the exact Jaccard of the planted pair's 5-word shingle sets, "
            "computed without the detector. Bins do not depend on the threshold."
        ),
    )
    bound = summary["bound"]
    lengths = bound["lengths"]
    bound_rows = tuple(
        (
            str(row["edits"]),
            str(row["guaranteed_length_words"]),
            *(f"{row['lower_bound_by_length'][str(n)]:.3f}" for n in lengths),
        )
        for row in bound["rows"]
    )
    arithmetic = Table(
        title="Why short texts with word edits fall below the threshold (arithmetic, not measured)",
        headers=(
            "Word edits k",
            "Length that guarantees the threshold (words)",
            *(f"Jaccard lower bound at {n} words" for n in lengths),
        ),
        rows=bound_rows,
        note=(
            f"With {bound['shingle']}-word shingles, a text of n words has S = n - "
            f"{bound['shingle'] - 1} shingles. One word edit changes at most five of them, so "
            "Jaccard is at least (S - 5k) / (S + 5k). For a threshold of 0.8 this needs "
            "S >= 45k, that is n >= 45k + 4 words. Most base items in this run fall in the "
            "short and medium length bins (base counts in the length table), so a single word "
            "edit can take a typical item below the threshold even when the detector works. "
            "Shingle size and threshold are the knobs; see docs/limitations.md."
        ),
    )
    return [conditional, bins, arithmetic]


def _scope_label(scope: Any) -> str:
    return {
        "cross": "cross-split",
        "cross_split": "cross-split",
        "within": "within-split",
        "within_split": "within-split",
    }.get(scope, "n/a")


def _finding_scopes(
    name: str, findings: list[dict[str, Any]], reports: dict[str, dict[str, Any]] | None
) -> list[str]:
    """Scope of each summary finding, read from the dataset's report (same order)."""
    if reports is None or name not in reports:
        return ["n/a"] * len(findings)
    report_findings = reports[name]["findings"]
    if len(report_findings) != len(findings):
        return ["n/a"] * len(findings)
    return [_scope_label(rf.get("details", {}).get("scope")) for rf in report_findings]


def findings_tables(
    e2: list[dict[str, Any]],
    e3: dict[str, Any] | None,
    e3_reports: dict[str, dict[str, Any]] | None = None,
) -> list[Table]:
    """The E2 and E3 tables for the README and the site.

    E3 rows carry the scope (cross-split or within-split) and the split pair, read from each
    dataset's report, so a within-split count is not read as a cross-split one.
    """
    tables: list[Table] = []
    if e2:
        rows = []
        for r in e2:
            acc = r["accuracy"]
            lit = r["literature"]
            rows.append(
                (
                    r["dataset"],
                    str(r["n_eval"]),
                    str(r["n_train"]),
                    f"{pct(acc['value'])} [{pct(acc['ci_low'])}, {pct(acc['ci_high'])}]",
                    pct(r["majority"]),
                    f"{pct(lit['value'], 1)} (reported)",
                    f"{r['gap_points']:+.1f}",
                    "yes" if r["flagged"] else "no",
                )
            )
        tables.append(
            Table(
                title="E2 hypothesis-only accuracy against reported literature",
                headers=(
                    "Dataset",
                    "Eval items",
                    "Train items",
                    "Accuracy [95% CI]",
                    "Majority baseline",
                    "Literature (reported)",
                    "Gap (points)",
                    "Flagged",
                ),
                rows=tuple(rows),
                note=(
                    f"Literature values: {LITERATURE_CITATION}. Checked against that table; "
                    "they are cited, not our measurements. Our classifier is TF-IDF with "
                    "logistic regression, not the models of the cited paper, so this tests the "
                    "size of a shortcut, not a replication."
                ),
            )
        )
    if e3 is not None:
        rows3: list[tuple[str, ...]] = []
        for d in e3["datasets"]:
            if d["status"] != "ok":
                rows3.append(
                    (d["name"], "error", "n/a", "n/a", "n/a", str(d.get("error") or ""), "n/a")
                )
                continue
            scopes = _finding_scopes(d["name"], d["findings"], e3_reports)
            for f, scope in zip(d["findings"], scopes, strict=True):
                rows3.append(
                    (
                        d["name"],
                        f["check"],
                        scope,
                        " / ".join(f["splits"]),
                        f["severity"],
                        f"{f['count']} / {f['denominator']}",
                        f"{pct(f['rate'])} [{pct(f['ci_low'])}, {pct(f['ci_high'])}]",
                    )
                )
        tables.append(
            Table(
                title="E3 leakage scan",
                headers=(
                    "Dataset",
                    "Check",
                    "Scope",
                    "Split pair",
                    "Severity",
                    "Count / n",
                    "Rate [95% CI]",
                ),
                rows=tuple(rows3),
                note=(
                    "A finding means the lexical check matched. Shared templates can match "
                    "without contamination. Near findings list both splits; the scope column "
                    "separates cross-split counts from within-split counts. Example pairs are in "
                    "results/e3/."
                ),
            )
        )
    return tables


def tables_markdown(tables: Sequence[Table]) -> str:
    """Markdown for a list of tables: a heading, the table and its note per table."""
    parts: list[str] = []
    for t in tables:
        parts.append(f"### {t.title}")
        parts.append("")
        parts.append(md_table(t.headers, t.rows))
        if t.note:
            parts.append("")
            parts.append(t.note)
        parts.append("")
    return "\n".join(parts).rstrip()


def tables_html(tables: Sequence[Table]) -> str:
    """HTML for a list of tables, with escaped cells."""
    parts: list[str] = []
    for t in tables:
        parts.append(f"<h3>{html.escape(t.title)}</h3>")
        parts.append(html_table(t.headers, t.rows))
        if t.note:
            parts.append(f'<p class="note">{html.escape(t.note)}</p>')
    return "\n".join(parts)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A Markdown table; cell text is escaped."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(str(c)) for c in row) + " |")
    return "\n".join(lines)


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """An HTML table with escaped cells."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        '<div style="overflow-x:auto"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def calibration_block(calib: dict[str, Any] | None, summary: dict[str, Any] | None = None) -> str:
    """README block for E1: a source line, the synthetic label, and the tables."""
    if calib is None:
        return NO_RESULTS
    header = (
        f"Source: results/e1/calibration.json (mode {calib['mode']}), with the summary in "
        "results/e1/summary.json. Pre-registered in docs/prereg.md, E1. The commit and working "
        "tree of each run are in the provenance table at the end of the findings section."
    )
    return "\n\n".join(
        [header, f"**{SYNTHETIC_EDITS_NOTE}**", tables_markdown(calibration_tables(calib, summary))]
    )


def _provenance_row(name: str, env: dict[str, Any] | None) -> tuple[str, ...]:
    """One row of the provenance table, read from an experiment's env.json."""
    if env is None:
        return (name, "n/a", "no env.json", "n/a")
    sha = env.get("git_sha")
    commit = str(sha)[:10] if sha else "n/a"
    dirty = env.get("git_dirty")
    if dirty is None:
        state = "git unavailable"
    elif dirty is False:
        state = "clean tree (no uncommitted changes outside results/)"
    else:
        paths = env.get("git_dirty_paths")
        if paths is None:
            state = "uncommitted changes; the paths were not recorded (envinfo before the change)"
        else:
            shown = ", ".join(paths[:5]) + (" and more" if len(paths) > 5 else "")
            state = f"uncommitted changes: {shown}"
    digest = env.get("git_diff_sha256")
    if digest:
        state += f"; diff hash {str(digest)[:12]}"
    return (name, commit, state, str(env.get("utc_time") or "n/a"))


def provenance_table(envs: Sequence[tuple[str, dict[str, Any] | None]]) -> Table:
    """Commit and working-tree state of each experiment, from its own env.json."""
    return Table(
        title="Provenance of the committed results (from each env.json)",
        headers=("Experiment", "Commit", "Working tree at run", "Recorded (UTC)"),
        rows=tuple(_provenance_row(name, env) for name, env in envs),
        note=(
            "Each row is read from that experiment's env.json. A run from a dirty tree is not "
            "reproducible from the commit alone. The code differences are in docs/deviations.md."
        ),
    )


def findings_block(
    e2: list[dict[str, Any]],
    e3: dict[str, Any] | None,
    e3_reports: dict[str, dict[str, Any]] | None,
    envs: Sequence[tuple[str, dict[str, Any] | None]],
) -> str:
    """README block for E2 and E3 with their provenance, or the no-results text."""
    tables = findings_tables(e2, e3, e3_reports)
    if not tables:
        return NO_RESULTS
    header = (
        "Source: results/e2/*.json, results/e3/summary.json and results/e3/<dataset>.json "
        "(full mode)."
    )
    return "\n\n".join([header, tables_markdown([*tables, provenance_table(envs)])])


def render_index_html(
    title: str,
    calib: dict[str, Any] | None,
    e2: list[dict[str, Any]],
    e3: dict[str, Any] | None,
    e3_links: Sequence[tuple[str, str, str]],
    summary: dict[str, Any] | None = None,
    e3_reports: dict[str, dict[str, Any]] | None = None,
    envs: Sequence[tuple[str, dict[str, Any] | None]] = (),
) -> str:
    """The static site index: E1, E2 and E3 tables with provenance, and per-dataset links."""
    e1_html = (
        tables_html(calibration_tables(calib, summary))
        if calib
        else f"<p>{html.escape(NO_RESULTS)}</p>"
    )
    findings = findings_tables(e2, e3, e3_reports)
    if findings:
        findings.append(provenance_table(envs))
        e2_html = tables_html(findings)
    else:
        e2_html = f"<p>{html.escape(NO_RESULTS)}</p>"
    links = (
        "".join(
            f'<li><a href="{html.escape(href)}">{html.escape(name)}</a> ({html.escape(status)})</li>'
            for name, href, status in e3_links
        )
        or "<li>No E3 audit reports yet.</li>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#ffffff; --fg:#1d1f23; --muted:#5b616b; --line:#d8dce3; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1115; --fg:#e6e8ec; --muted:#9aa3b2; --line:#2c313a; }} }}
body {{ background: var(--bg); color: var(--fg); margin: 0; padding: 16px;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1000px; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 8px 0 16px; }}
th, td {{ text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line); }}
th {{ color: var(--muted); }}
.note {{ color: var(--muted); }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="note">Generated from committed JSON in <code>results/</code>. Numbers are not typed by hand.
Smoke runs are never shown here.</p>
<h2>E1. Detector calibration</h2>
<p class="note">{html.escape(SYNTHETIC_EDITS_NOTE)}</p>
{e1_html}
<h2>E2 and E3. Findings</h2>
{e2_html}
<h2>Per-dataset audit reports</h2>
<ul>{links}</ul>
</main>
</body>
</html>
"""
