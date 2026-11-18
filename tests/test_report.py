"""Tests for report.py: renderers, fail-on logic, Wilson intervals and the committed schema."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from benchlint import report
from benchlint.schema import (
    AuditReport,
    DatasetSpec,
    ExamplePair,
    Finding,
    ShortcutResult,
    SourceSpec,
)

REPO = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO / "schemas" / "audit_report.schema.json"


def _spec(name: str = "toy") -> DatasetSpec:
    return DatasetSpec(
        name=name,
        source=SourceSpec(
            kind="jsonl",
            id_or_path="data",
            config=None,
            revision=None,
            split_map={"train": "train", "test": "test"},
        ),
        text_fields=["text"],
        label_field="label",
        id_field="id",
        max_rows_per_split=None,
        seed=0,
    )


def _finding(
    severity: str = "high",
    count: int = 2,
    denominator: int = 10,
    check: str = "exact_across",
    text: str = "a | b <script>alert(1)</script>",
) -> Finding:
    return Finding(
        check=check,
        severity=severity,  # type: ignore[arg-type]
        splits=["train", "test"],
        count=count,
        denominator=denominator,
        rate=count / denominator if denominator else 0.0,
        examples=[
            ExamplePair(
                a_id="train-1",
                b_id="test-2",
                a_split="train",
                b_split="test",
                similarity=1.0,
                a_text=text,
                b_text=text,
            )
        ],
        details={"n_groups": count, "pairs": [["train-1", "test-2"]]},
    )


def _shortcut(flagged: bool = True) -> ShortcutResult:
    return ShortcutResult(
        view="headline_only",
        metric="accuracy",
        value=0.9,
        ci_low=0.85,
        ci_high=0.95,
        majority=0.25,
        chance=0.25,
        n_train=100,
        n_test=50,
        p_value=0.001,
        flagged=flagged,
    )


def _report(
    findings: list[Finding] | None = None,
    shortcuts: list[ShortcutResult] | None = None,
    dataset: str = "toy",
) -> AuditReport:
    return AuditReport(
        dataset=dataset,
        spec=_spec(dataset),
        findings=findings if findings is not None else [_finding()],
        shortcuts=shortcuts or [],
        env_info={"utc_time": "2026-01-01T00:00:00+00:00", "python_version": "3.11.0"},
        runtime_s=1.25,
    )


class _TagCollector(HTMLParser):
    """Collects start-tag names. Raises if the markup is not parseable."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)


# --- Wilson interval -----------------------------------------------------------------------


def test_wilson_interval_zero_successes_has_zero_lower_bound() -> None:
    lo, hi = report.wilson_interval(0, 10)
    assert lo == 0.0
    assert 0.27 < hi < 0.29


def test_wilson_interval_is_symmetric_at_one_half() -> None:
    lo, hi = report.wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0, abs=1e-9)
    assert 0.2 < lo < 0.3


def test_wilson_interval_zero_denominator_is_empty() -> None:
    assert report.wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_contains_point_estimate() -> None:
    for k, n in [(1, 7), (3, 40), (39, 40), (100, 1000)]:
        lo, hi = report.wilson_interval(k, n)
        assert lo <= k / n <= hi


# --- severity and fail-on ------------------------------------------------------------------


def test_severity_rank_orders_levels() -> None:
    ranks = [report.severity_rank(s) for s in ("info", "low", "medium", "high")]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == 4


def test_severity_rank_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        report.severity_rank("critical")


@pytest.mark.parametrize(
    ("severity", "fail_on", "expected"),
    [
        ("high", "high", True),
        ("medium", "high", False),
        ("medium", "medium", True),
        ("low", "medium", False),
        ("info", "info", True),
        ("info", "low", False),
    ],
)
def test_exceeds_fail_on_by_severity(severity: str, fail_on: str, expected: bool) -> None:
    rep = _report([_finding(severity=severity, count=1)], shortcuts=[])
    assert report.exceeds_fail_on(rep, fail_on) is expected


def test_zero_count_finding_never_fails() -> None:
    rep = _report([_finding(severity="high", count=0)])
    assert report.exceeds_fail_on(rep, "info") is False


def test_flagged_shortcut_counts_as_high() -> None:
    rep = _report([], shortcuts=[_shortcut(flagged=True)])
    assert report.exceeds_fail_on(rep, "high") is True
    assert report.exceeds_fail_on(rep, "info") is True


def test_unflagged_shortcut_does_not_fail() -> None:
    rep = _report([], shortcuts=[_shortcut(flagged=False)])
    assert report.exceeds_fail_on(rep, "high") is False


def test_exceeds_fail_on_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        report.exceeds_fail_on(_report(), "critical")


# --- synthetic labelling -------------------------------------------------------------------


def test_is_synthetic_reads_dataset_name() -> None:
    assert report.is_synthetic(_report(dataset="SYNTHETIC-demo")) is True
    assert report.is_synthetic(_report(dataset="gsm8k")) is False


# --- renderers -------------------------------------------------------------------------------


def test_render_json_validates_against_model_and_schema_keys() -> None:
    rep = _report([_finding()], shortcuts=[_shortcut()])
    text = report.render_json(rep)
    parsed = AuditReport.model_validate_json(text)
    assert parsed == rep
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(json.loads(text)) == set(schema["properties"])
    for key in schema["required"]:
        assert key in json.loads(text)


def test_render_markdown_has_banner_only_for_synthetic() -> None:
    synthetic = report.render_markdown(_report(dataset="SYNTHETIC-demo"))
    assert "SYNTHETIC" in synthetic.splitlines()[2]
    plain = report.render_markdown(_report(dataset="gsm8k"))
    assert "SYNTHETIC" not in plain


def test_render_markdown_escapes_pipes_in_cells() -> None:
    md = report.render_markdown(_report([_finding(text="x | y")]))
    assert "x \\| y" in md
    table_rows = [line for line in md.splitlines() if line.startswith("| train-1 |")]
    assert table_rows
    assert all(line.count("|") - line.count("\\|") == 8 for line in table_rows)


def test_render_markdown_shows_rate_with_interval_and_denominator() -> None:
    md = report.render_markdown(_report([_finding(count=2, denominator=10)]))
    assert "2 / 10" in md  # count / n column
    assert "(n=10)" in md
    assert "Wilson" in md


def test_render_html_is_parseable_and_has_structure() -> None:
    rep = _report(
        [_finding(severity="high"), _finding(severity="medium", check="near")], [_shortcut()]
    )
    page = report.render_html(rep)
    assert page.lstrip().lower().startswith("<!doctype html>")
    collector = _TagCollector()
    collector.feed(page)
    assert "details" in collector.tags
    assert "table" in collector.tags
    assert "style" in collector.tags
    assert "sev-high" in page
    assert "sev-medium" in page
    assert "prefers-color-scheme: dark" in page


def test_render_html_escapes_dataset_text() -> None:
    page = report.render_html(_report([_finding()]))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_render_html_synthetic_banner() -> None:
    assert "SYNTHETIC" in report.render_html(_report(dataset="SYNTHETIC-demo"))
    assert "SYNTHETIC" not in report.render_html(_report(dataset="gsm8k"))


def test_render_html_handles_empty_report() -> None:
    page = report.render_html(_report(findings=[], shortcuts=[]))
    assert "No findings were reported." in page
    assert "No shortcut baselines were run." in page


def test_write_reports_writes_three_formats(tmp_path: Path) -> None:
    rep = _report([_finding()], [_shortcut()])
    paths = report.write_reports(rep, tmp_path / "out")
    assert [p.name for p in paths] == ["report.json", "report.md", "report.html"]
    assert all(p.is_file() for p in paths)
    AuditReport.model_validate_json(paths[0].read_text(encoding="utf-8"))


def test_write_reports_subset_and_unknown_format(tmp_path: Path) -> None:
    paths = report.write_reports(_report(), tmp_path, formats=("json",))
    assert [p.name for p in paths] == ["report.json"]
    with pytest.raises(ValueError):
        report.write_reports(_report(), tmp_path, formats=("pdf",))


def test_collect_env_info_has_provenance_keys() -> None:
    info: dict[str, Any] = report.collect_env_info()
    for key in ("utc_time", "python_version", "os_name", "machine", "packages"):
        assert key in info


# --- committed schema ------------------------------------------------------------------------


def test_committed_schema_equals_generated_schema() -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == AuditReport.model_json_schema()


def test_committed_schema_file_is_stable_text() -> None:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    regenerated = json.dumps(AuditReport.model_json_schema(), indent=2, sort_keys=True) + "\n"
    assert text == regenerated
