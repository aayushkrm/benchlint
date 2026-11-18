"""Report writers: JSON, Markdown and static HTML (SARIF is a stretch goal).

Every rate is shown with its denominator and a 95% Wilson interval. Synthetic runs (the
dataset name contains SYNTHETIC) carry a banner in every format.

The Markdown and HTML templates are kept inline so the package needs no data files. Their long
template lines are exempt from E501 for this module only (``# ruff: noqa: E501``).
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from jinja2 import Environment

from benchlint import __version__
from benchlint.schema import AuditReport, Finding, ShortcutResult

SEVERITIES: tuple[str, ...] = ("info", "low", "medium", "high")
FAIL_ON_NONE = "none"
"""``--fail-on none``: never exit 1 for findings (exit 0 unless there is a usage or runtime error)."""
SHORTCUT_SEVERITY = "high"
FORMATS: tuple[str, ...] = ("json", "md", "html")
_REPO_DIR = Path(__file__).resolve().parents[2]
_SYNTHETIC_BANNER = (
    "SYNTHETIC: this report was produced from generated demo data with planted leaks. "
    "No real dataset was audited."
)


def annotate_intervals(findings: Sequence[Finding]) -> None:
    """Store the 95% Wilson interval of each finding's rate in its ``details``.

    Keys ``rate_ci_low`` and ``rate_ci_high`` make the JSON report carry the same intervals
    the Markdown and HTML reports show.
    """
    for f in findings:
        lo, hi = wilson_interval(f.count, f.denominator)
        f.details["rate_ci_low"] = round(lo, 6)
        f.details["rate_ci_high"] = round(hi, 6)


def severity_rank(level: str) -> int:
    """Position of ``level`` in ``SEVERITIES`` (higher is more severe).

    Raises ``ValueError`` for an unknown level.
    """
    if level not in SEVERITIES:
        raise ValueError(f"unknown severity {level!r}; expected one of {', '.join(SEVERITIES)}")
    return SEVERITIES.index(level)


def findings_exceed(
    findings: Sequence[Finding], shortcuts: Sequence[ShortcutResult], fail_on: str
) -> bool:
    """True if a finding with count > 0, or a flagged shortcut, is at or above ``fail_on``."""
    if fail_on == FAIL_ON_NONE:
        return False
    threshold = severity_rank(fail_on)
    for f in findings:
        if f.count > 0 and severity_rank(f.severity) >= threshold:
            return True
    if severity_rank(SHORTCUT_SEVERITY) >= threshold:
        return any(s.flagged for s in shortcuts)
    return False


def exceeds_fail_on(report: AuditReport, fail_on: str) -> bool:
    """True if any finding or flagged shortcut is at or above the ``fail_on`` severity.

    ``fail_on`` is one of ``info``, ``low``, ``medium``, ``high``. Drives exit code 1. A finding
    with count 0 never fails. A flagged shortcut counts as ``SHORTCUT_SEVERITY``.
    """
    return findings_exceed(report.findings, report.shortcuts, fail_on)


def is_synthetic(report: AuditReport) -> bool:
    """True if the report is labelled SYNTHETIC (by dataset or spec name)."""
    return "SYNTHETIC" in report.dataset.upper() or "SYNTHETIC" in report.spec.name.upper()


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Returns ``(0.0, 0.0)`` when ``n`` is 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def collect_env_info() -> dict[str, Any]:
    """Provenance dict for a report. Uses ``scripts/envinfo.py`` when it is present in the repo
    checkout; otherwise a minimal dict with the same key names where they apply.
    """
    script = _REPO_DIR / "scripts" / "envinfo.py"
    if script.is_file():
        spec = importlib.util.spec_from_file_location("benchlint_envinfo", script)
        if spec is not None and spec.loader is not None:
            module: ModuleType = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            collected: dict[str, Any] = module.collect()
            return collected
    import platform
    from datetime import UTC, datetime

    return {
        "git_sha": None,
        "git_dirty": None,
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_model": None,
        "cpu_flags": [],
        "cores_physical": None,
        "cores_logical": None,
        "ram_bytes": None,
        "os_name": platform.system(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "packages": {"benchlint": __version__},
        "note": "scripts/envinfo.py not found; minimal provenance only",
    }


@dataclass(frozen=True)
class ExampleView:
    """One example pair, ready for display."""

    a_id: str
    b_id: str
    a_split: str
    b_split: str
    similarity: str
    a_text: str
    b_text: str


@dataclass(frozen=True)
class FindingView:
    """One finding, ready for display."""

    check: str
    severity: str
    splits: str
    count_text: str
    rate_text: str
    details_json: str
    examples: list[ExampleView] = field(default_factory=list)


@dataclass(frozen=True)
class ShortcutView:
    """One shortcut baseline, ready for display."""

    view: str
    metric: str
    value_text: str
    majority_text: str
    chance_text: str
    n_text: str
    p_text: str
    flagged: bool


@dataclass(frozen=True)
class ReportView:
    """Everything the templates need, precomputed from an AuditReport."""

    title: str
    dataset: str
    synthetic: bool
    banner: str
    version: str
    schema_version: str
    runtime_s: str
    n_findings: int
    max_severity: str
    n_flagged: int
    splits: str
    findings: list[FindingView]
    shortcuts: list[ShortcutView]
    env: list[tuple[str, str]]


def _cell(text: str) -> str:
    """Escape a value for a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _details_json(details: dict[str, Any]) -> str:
    return json.dumps(details, indent=2, sort_keys=True, default=str)


def _build_view(report: AuditReport) -> ReportView:
    synthetic = is_synthetic(report)
    findings: list[FindingView] = []
    for f in report.findings:
        lo, hi = wilson_interval(f.count, f.denominator)
        examples = [
            ExampleView(
                a_id=e.a_id,
                b_id=e.b_id,
                a_split=e.a_split,
                b_split=e.b_split,
                similarity=f"{e.similarity:.3f}",
                a_text=e.a_text,
                b_text=e.b_text,
            )
            for e in f.examples
        ]
        findings.append(
            FindingView(
                check=f.check,
                severity=f.severity,
                splits=", ".join(f.splits),
                count_text=f"{f.count} / {f.denominator}",
                rate_text=f"{_pct(f.rate)} [{_pct(lo)}, {_pct(hi)}] (n={f.denominator})",
                details_json=_details_json(f.details),
                examples=examples,
            )
        )
    shortcuts = [
        ShortcutView(
            view=s.view,
            metric=s.metric,
            value_text=f"{s.value:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]",
            majority_text=f"{s.majority:.3f}",
            chance_text=f"{s.chance:.3f}",
            n_text=f"{s.n_train} / {s.n_test}",
            p_text="n/a" if s.p_value is None else f"{s.p_value:.4f}",
            flagged=s.flagged,
        )
        for s in report.shortcuts
    ]
    max_sev = "none"
    for f in report.findings:
        if f.count > 0 and (
            max_sev == "none" or severity_rank(f.severity) > severity_rank(max_sev)
        ):
            max_sev = f.severity
    env_keys = ("git_sha", "utc_time", "python_version", "os_version", "cpu_model", "machine")
    env = [(k, str(report.env_info.get(k, "n/a"))) for k in env_keys]
    return ReportView(
        title=f"benchlint report: {report.dataset}",
        dataset=report.dataset,
        synthetic=synthetic,
        banner=_SYNTHETIC_BANNER,
        version=__version__,
        schema_version=report.schema_version,
        runtime_s=f"{report.runtime_s:.2f}",
        n_findings=sum(1 for f in report.findings if f.count > 0),
        max_severity=max_sev,
        n_flagged=sum(1 for s in report.shortcuts if s.flagged),
        splits=", ".join(sorted({s for f in report.findings for s in f.splits})) or "n/a",
        findings=findings,
        shortcuts=shortcuts,
        env=env,
    )


def render_json(report: AuditReport) -> str:
    """Render ``report`` as JSON that validates against ``schemas/audit_report.schema.json``."""
    return report.model_dump_json(indent=2) + "\n"


_MD_TEMPLATE = """# {{ v.title }}

{% if v.synthetic %}> **{{ v.banner }}**

{% endif %}Schema version {{ v.schema_version }} · benchlint {{ v.version }} · runtime {{ v.runtime_s }} s

## Summary

- Findings with at least one match: {{ v.n_findings }} (highest severity: {{ v.max_severity }})
- Shortcut baselines flagged: {{ v.n_flagged }}
- Splits involved in findings: {{ v.splits }}

## Findings

| Check | Severity | Splits | Count / n | Rate [95% Wilson CI] |
|---|---|---|---|---|
{% for f in v.findings %}| {{ f.check }} | {{ f.severity }} | {{ f.splits }} | {{ f.count_text }} | {{ f.rate_text }} |
{% endfor %}
{% for f in v.findings %}{% if f.examples or f.details_json != "{}" %}<details><summary>{{ f.check }}: details and examples</summary>

```json
{{ f.details_json }}
```
{% if f.examples %}
| A id | B id | A split | B split | Similarity | A text | B text |
|---|---|---|---|---|---|---|
{% for e in f.examples %}| {{ e.a_id }} | {{ e.b_id }} | {{ e.a_split }} | {{ e.b_split }} | {{ e.similarity }} | {{ e.a_text }} | {{ e.b_text }} |
{% endfor %}{% endif %}
</details>

{% endif %}{% endfor %}
## Shortcut baselines

{% if v.shortcuts %}| View | Metric | Value [95% CI] | Majority | Chance | n train / test | p-value | Flagged |
|---|---|---|---|---|---|---|---|
{% for s in v.shortcuts %}| {{ s.view }} | {{ s.metric }} | {{ s.value_text }} | {{ s.majority_text }} | {{ s.chance_text }} | {{ s.n_text }} | {{ s.p_text }} | {{ "yes" if s.flagged else "no" }} |
{% endfor %}{% else %}No shortcut baselines were run.
{% endif %}
## Environment

{% for k, val in v.env %}- {{ k }}: {{ val }}
{% endfor %}
## Limits

- Matching is lexical. Paraphrases are not detected.
- A flag is a prompt to investigate, not proof of a shortcut or a leak.
- Each rate carries its denominator and a 95% Wilson interval.
"""


def render_markdown(report: AuditReport) -> str:
    """Render ``report`` as Markdown. Synthetic runs must carry a SYNTHETIC label."""
    env = Environment(autoescape=False)
    # Table cells are escaped in the view so the template can print them directly.
    view = _escape_view_for_md(_build_view(report))
    return env.from_string(_MD_TEMPLATE).render(v=view).rstrip() + "\n"


def _escape_view_for_md(view: ReportView) -> ReportView:
    findings = [
        FindingView(
            check=_cell(f.check),
            severity=f.severity,
            splits=_cell(f.splits),
            count_text=f.count_text,
            rate_text=f.rate_text,
            details_json=f.details_json,
            examples=[
                ExampleView(
                    a_id=_cell(e.a_id),
                    b_id=_cell(e.b_id),
                    a_split=_cell(e.a_split),
                    b_split=_cell(e.b_split),
                    similarity=e.similarity,
                    a_text=_cell(e.a_text),
                    b_text=_cell(e.b_text),
                )
                for e in f.examples
            ],
        )
        for f in view.findings
    ]
    shortcuts = [
        ShortcutView(
            view=_cell(s.view),
            metric=_cell(s.metric),
            value_text=s.value_text,
            majority_text=s.majority_text,
            chance_text=s.chance_text,
            n_text=s.n_text,
            p_text=s.p_text,
            flagged=s.flagged,
        )
        for s in view.shortcuts
    ]
    return ReportView(
        title=view.title,
        dataset=view.dataset,
        synthetic=view.synthetic,
        banner=view.banner,
        version=view.version,
        schema_version=view.schema_version,
        runtime_s=view.runtime_s,
        n_findings=view.n_findings,
        max_severity=view.max_severity,
        n_flagged=view.n_flagged,
        splits=view.splits,
        findings=findings,
        shortcuts=shortcuts,
        env=view.env,
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ v.title }}</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1d1f23; --muted: #5b616b; --line: #d8dce3; --card: #f5f7fa;
  --sev-info: #6b7280; --sev-low: #2563eb; --sev-medium: #b45309; --sev-high: #b91c1c;
  --flag: #b91c1c; --ok: #15803d;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1115; --fg: #e6e8ec; --muted: #9aa3b2; --line: #2c313a; --card: #181c23;
    --sev-info: #9ca3af; --sev-low: #60a5fa; --sev-medium: #fbbf24; --sev-high: #f87171;
    --flag: #f87171; --ok: #4ade80;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1115; --fg: #e6e8ec; --muted: #9aa3b2; --line: #2c313a; --card: #181c23;
  --sev-info: #9ca3af; --sev-low: #60a5fa; --sev-medium: #fbbf24; --sev-high: #f87171;
  --flag: #f87171; --ok: #4ade80;
}
body { background: var(--bg); color: var(--fg); margin: 0; padding: 16px;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0.5rem 0; overflow-wrap: anywhere; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid var(--line); }
.banner { border: 2px solid var(--sev-medium); padding: 10px 12px; border-radius: 6px;
  font-weight: 600; background: var(--card); }
.meta { color: var(--muted); font-size: 0.9rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px;
  margin: 12px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; }
.card .k { color: var(--muted); font-size: 0.8rem; }
.card .v { font-size: 1.2rem; font-weight: 600; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px;
  border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.sev { display: inline-block; padding: 1px 8px; border-radius: 999px; color: #fff;
  font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
.sev-info { background: var(--sev-info); }
.sev-low { background: var(--sev-low); }
.sev-medium { background: var(--sev-medium); }
.sev-high { background: var(--sev-high); }
.flag { color: var(--flag); font-weight: 700; }
.ok { color: var(--ok); }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--muted); }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px; overflow-x: auto; font-size: 0.82rem; }
td.text { max-width: 320px; overflow-wrap: anywhere; }
footer { color: var(--muted); font-size: 0.85rem; margin: 2rem 0 1rem; }
</style>
</head>
<body>
<main>
<h1>{{ v.title }}</h1>
{% if v.synthetic %}<div class="banner" role="note">{{ v.banner }}</div>{% endif %}
<p class="meta">Schema version {{ v.schema_version }} · benchlint {{ v.version }} · runtime {{ v.runtime_s }} s</p>

<div class="cards">
  <div class="card"><div class="k">Findings with matches</div><div class="v">{{ v.n_findings }}</div></div>
  <div class="card"><div class="k">Highest severity</div><div class="v">
    <span class="sev sev-{{ v.max_severity }}">{{ v.max_severity }}</span></div></div>
  <div class="card"><div class="k">Shortcuts flagged</div><div class="v {% if v.n_flagged %}flag{% else %}ok{% endif %}">{{ v.n_flagged }}</div></div>
  <div class="card"><div class="k">Splits in findings</div><div class="v" style="font-size:1rem">{{ v.splits }}</div></div>
</div>

<h2>Findings</h2>
{% if v.findings %}
<div class="table-wrap"><table>
<thead><tr><th>Check</th><th>Severity</th><th>Splits</th><th>Count / n</th><th>Rate [95% Wilson CI]</th></tr></thead>
<tbody>
{% for f in v.findings %}
<tr>
  <td>{{ f.check }}</td>
  <td><span class="sev sev-{{ f.severity }}">{{ f.severity }}</span></td>
  <td>{{ f.splits }}</td>
  <td class="num">{{ f.count_text }}</td>
  <td class="num">{{ f.rate_text }}</td>
</tr>
{% endfor %}
</tbody></table></div>
{% for f in v.findings %}
<details>
  <summary>{{ f.check }}: details{% if f.examples %} and {{ f.examples|length }} example pair(s){% endif %}</summary>
  <pre>{{ f.details_json }}</pre>
  {% if f.examples %}
  <div class="table-wrap"><table>
  <thead><tr><th>A id</th><th>B id</th><th>A split</th><th>B split</th><th>Similarity</th><th>A text</th><th>B text</th></tr></thead>
  <tbody>
  {% for e in f.examples %}
  <tr>
    <td>{{ e.a_id }}</td><td>{{ e.b_id }}</td><td>{{ e.a_split }}</td><td>{{ e.b_split }}</td>
    <td class="num">{{ e.similarity }}</td>
    <td class="text">{{ e.a_text }}</td><td class="text">{{ e.b_text }}</td>
  </tr>
  {% endfor %}
  </tbody></table></div>
  {% endif %}
</details>
{% endfor %}
{% else %}
<p class="ok">No findings were reported.</p>
{% endif %}

<h2>Shortcut baselines</h2>
{% if v.shortcuts %}
<div class="table-wrap"><table>
<thead><tr><th>View</th><th>Metric</th><th>Value [95% CI]</th><th>Majority</th><th>Chance</th><th>n train / test</th><th>p-value</th><th>Flagged</th></tr></thead>
<tbody>
{% for s in v.shortcuts %}
<tr>
  <td>{{ s.view }}</td><td>{{ s.metric }}</td><td class="num">{{ s.value_text }}</td>
  <td class="num">{{ s.majority_text }}</td><td class="num">{{ s.chance_text }}</td>
  <td class="num">{{ s.n_text }}</td><td class="num">{{ s.p_text }}</td>
  <td>{% if s.flagged %}<span class="flag">yes</span>{% else %}no{% endif %}</td>
</tr>
{% endfor %}
</tbody></table></div>
{% else %}
<p class="meta">No shortcut baselines were run.</p>
{% endif %}

<h2>Environment</h2>
<table><tbody>
{% for k, val in v.env %}<tr><th>{{ k }}</th><td>{{ val }}</td></tr>
{% endfor %}</tbody></table>

<h2>Limits</h2>
<ul>
  <li>Matching is lexical. Paraphrases are not detected.</li>
  <li>A flag is a prompt to investigate, not proof of a shortcut or a leak.</li>
  <li>Each rate carries its denominator and a 95% Wilson interval.</li>
</ul>
<footer>Generated by benchlint {{ v.version }}. Report schema: schemas/audit_report.schema.json.</footer>
</main>
</body>
</html>
"""


def render_html(report: AuditReport) -> str:
    """Render ``report`` as a self-contained static HTML page (Jinja2 template).

    All dataset text is HTML-escaped. The page has inline CSS only, supports light and dark
    colour schemes, and collapses examples in ``<details>``.
    """
    env = Environment(autoescape=True)
    view = _build_view(report)
    return env.from_string(_HTML_TEMPLATE).render(v=view)


def write_reports(
    report: AuditReport,
    out_dir: Path,
    formats: Sequence[str] = ("json", "md", "html"),
) -> list[Path]:
    """Write one file per requested format into ``out_dir`` and return the paths written.

    Files are named ``report.json``, ``report.md`` and ``report.html``. Unknown formats raise
    ``ValueError``.
    """
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown report format(s): {', '.join(unknown)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {"json": render_json, "md": render_markdown, "html": render_html}
    suffix = {"json": "json", "md": "md", "html": "html"}
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"report.{suffix[fmt]}"
        path.write_text(writers[fmt](report), encoding="utf-8")
        written.append(path)
    return written


__all__ = [
    "FORMATS",
    "SEVERITIES",
    "SHORTCUT_SEVERITY",
    "collect_env_info",
    "exceeds_fail_on",
    "findings_exceed",
    "annotate_intervals",
    "is_synthetic",
    "render_html",
    "render_json",
    "render_markdown",
    "severity_rank",
    "wilson_interval",
    "write_reports",
]
