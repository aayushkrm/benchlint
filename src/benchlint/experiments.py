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
    "Gururangan et al. (2018), Annotation Artifacts in Natural Language Inference Data, NAACL"
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


def calibration_tables(calib: dict[str, Any] | None) -> list[Table]:
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
    return [
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
            note=f"Random pairs of clean controls; flagged at threshold {threshold}.",
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


def findings_tables(e2: list[dict[str, Any]], e3: dict[str, Any] | None) -> list[Table]:
    """The E2 and E3 tables for the README and the site."""
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
                    f"{pct(lit['value'], 0)} (reported)",
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
                    f"Literature values: {LITERATURE_CITATION}. They are cited, not our "
                    "measurements. Our classifier is TF-IDF with logistic regression, not the "
                    "models of the cited paper, so this tests the size of a shortcut, not a "
                    "replication."
                ),
            )
        )
    if e3 is not None:
        rows3: list[tuple[str, ...]] = []
        for d in e3["datasets"]:
            if d["status"] != "ok":
                rows3.append((d["name"], "error", "n/a", str(d.get("error") or ""), "n/a"))
                continue
            for f in d["findings"]:
                rows3.append(
                    (
                        d["name"],
                        f["check"],
                        f["severity"],
                        f"{f['count']} / {f['denominator']}",
                        f"{pct(f['rate'])} [{pct(f['ci_low'])}, {pct(f['ci_high'])}]",
                    )
                )
        tables.append(
            Table(
                title="E3 leakage scan",
                headers=("Dataset", "Check", "Severity", "Count / n", "Rate [95% CI]"),
                rows=tuple(rows3),
                note=(
                    "A finding means the lexical check matched. Shared templates can match "
                    "without contamination. Example pairs are in `results/e3/`."
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


def calibration_block(calib: dict[str, Any] | None, env: dict[str, Any] | None) -> str:
    """README block for E1: a source line, the synthetic label, and the tables."""
    if calib is None:
        return NO_RESULTS
    sha = short_sha(env)
    header = (
        f"Source: `results/e1/calibration.json` (mode {calib['mode']}, git `{sha}`, environment "
        "in `results/e1/env.json`). Pre-registered in `docs/prereg.md`, E1."
    )
    return "\n\n".join(
        [header, f"**{SYNTHETIC_EDITS_NOTE}**", tables_markdown(calibration_tables(calib))]
    )


def findings_block(
    e2: list[dict[str, Any]], e3: dict[str, Any] | None, env: dict[str, Any] | None
) -> str:
    """README block for E2 and E3, or the no-results text."""
    tables = findings_tables(e2, e3)
    if not tables:
        return NO_RESULTS
    sha = short_sha(env)
    header = (
        "Source: `results/e2/*.json` and `results/e3/summary.json` (full mode, git "
        f"`{sha}`). Environment in `results/e2/env.json` and `results/e3/env.json`."
    )
    return "\n\n".join([header, tables_markdown(tables)])


def render_index_html(
    title: str,
    calib: dict[str, Any] | None,
    e2: list[dict[str, Any]],
    e3: dict[str, Any] | None,
    e3_links: Sequence[tuple[str, str, str]],
) -> str:
    """The static site index: E1, E2 and E3 tables, and links to per-dataset reports."""
    e1_html = (
        tables_html(calibration_tables(calib)) if calib else f"<p>{html.escape(NO_RESULTS)}</p>"
    )
    findings = tables_html(findings_tables(e2, e3))
    e2_html = findings if findings else f"<p>{html.escape(NO_RESULTS)}</p>"
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
