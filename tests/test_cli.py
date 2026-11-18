"""Tests for cli.py: exit codes (0 pass, 1 finding, 2 usage or runtime error), --json validity
against the committed schema, canary round trip and the end-to-end demo.

Most tests replace the checks and the loader with fakes, so they run without data. The canary
and demo tests use the real implementations. The demo test is skipped if a check still raises
NotImplementedError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchlint import cli
from benchlint.cli import app
from benchlint.demo_data import DemoManifest
from benchlint.schema import AuditReport, Finding, Record, ShortcutResult

REPO = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((REPO / "schemas" / "audit_report.schema.json").read_text("utf-8"))
runner = CliRunner()


def _spec_yaml(path: Path, *, layout: str = "audit", extra: str = "") -> Path:
    """Write an AuditSpec (layout 'audit') or a bare DatasetSpec (layout 'dataset')."""
    dataset = """name: toy
source:
  kind: jsonl
  id_or_path: data
  config: null
  revision: null
  split_map: {train: train, test: test}
text_fields: [text]
label_field: label
id_field: id
max_rows_per_split: null
seed: 0
"""
    if layout == "audit":
        body = "dataset:\n" + "".join(f"  {line}\n" for line in dataset.splitlines())
        body += extra
    else:
        body = dataset
    path.write_text(body, encoding="utf-8")
    return path


def _records() -> dict[str, list[Record]]:
    return {
        "train": [
            Record(id=f"train-{i}", split="train", fields={"text": f"t{i}"}, label="a")
            for i in range(3)
        ],
        "test": [
            Record(id=f"test-{i}", split="test", fields={"text": f"u{i}"}, label="b")
            for i in range(3)
        ],
    }


def _finding(
    severity: str = "high",
    count: int = 1,
    check: str = "exact_across",
    splits: tuple[str, ...] = ("train", "test"),
) -> Finding:
    return Finding(
        check=check,
        severity=severity,  # type: ignore[arg-type]
        splits=list(splits),
        count=count,
        denominator=3,
        rate=count / 3,
        examples=[],
        details={},
    )


def _shortcut(flagged: bool) -> ShortcutResult:
    return ShortcutResult(
        view="headline_only",
        metric="accuracy",
        value=0.9,
        ci_low=0.85,
        ci_high=0.95,
        majority=0.5,
        chance=0.5,
        n_train=3,
        n_test=3,
        p_value=0.01,
        flagged=flagged,
    )


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the loader and every check with configurable fakes. Returns the knobs."""
    knobs: dict[str, Any] = {
        "exact": [],
        "near": [],
        "labels": [],
        "shortcuts": [],
        "ngram": None,
        "raise": None,
    }

    def _maybe_raise() -> None:
        if knobs["raise"] is not None:
            raise knobs["raise"]

    def load_detailed(spec: Any, base_dir: Any = None) -> tuple[dict[str, list[Record]], list]:
        _maybe_raise()
        return _records(), []

    def load(spec: Any, base_dir: Any = None) -> dict[str, list[Record]]:
        _maybe_raise()
        return _records()

    def exact(records: Any, text_fields: Any, **kw: Any) -> list[Finding]:
        return list(knobs["exact"])

    def near(records: Any, text_fields: Any, config: Any, *, seed: int = 0) -> list[Finding]:
        return list(knobs["near"])

    def labels(records: Any, text_fields: Any, *, near: Any = None, **kw: Any) -> list[Finding]:
        return list(knobs["labels"])

    def shortcut(
        records: Any, config: Any, *, train_split: str, test_split: str, seed: int = 0
    ) -> list[ShortcutResult]:
        return list(knobs["shortcuts"])

    def ngram(eval_records: Any, reference_texts: Any, text_fields: Any, config: Any) -> Finding:
        list(reference_texts)
        return knobs["ngram"] or _finding("info", 0, check="ngram_overlap")

    monkeypatch.setattr(cli, "load_dataset", load)
    monkeypatch.setattr(cli, "load_dataset_detailed", load_detailed)
    monkeypatch.setattr(cli, "find_exact_duplicates", exact)
    monkeypatch.setattr(cli, "find_near_duplicates", near)
    monkeypatch.setattr(cli, "find_label_conflicts", labels)
    monkeypatch.setattr(cli, "run_shortcut_checks", shortcut)
    monkeypatch.setattr(cli, "find_ngram_overlap", ngram)
    return knobs


def _audit_json(stdout: str) -> AuditReport:
    """Parse audit JSON output and check its top-level keys against the committed schema."""
    report = AuditReport.model_validate_json(stdout)
    parsed = json.loads(stdout)
    assert set(parsed) == set(SCHEMA["properties"])
    for key in SCHEMA["required"]:
        assert key in parsed
    return report


# --- help and surface ----------------------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("audit", "check", "canary", "demo"):
        assert name in result.stdout
    check_help = runner.invoke(app, ["check", "--help"])
    for sub in ("exact", "near", "ngram", "shortcut"):
        assert sub in check_help.stdout


# --- audit exit codes ------------------------------------------------------------------------


def test_audit_exit_0_when_nothing_found(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "report.json").is_file()
    assert (tmp_path / "out" / "report.md").is_file()
    assert (tmp_path / "out" / "report.html").is_file()


def test_audit_exit_1_on_high_finding(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["exact"] = [_finding("high", count=2)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


@pytest.mark.parametrize(
    ("severity", "fail_on", "exit_code"),
    [
        ("medium", "high", 0),
        ("medium", "medium", 1),
        ("low", "medium", 0),
        ("low", "low", 1),
        ("info", "high", 0),
        ("info", "info", 1),
    ],
)
def test_audit_fail_on_threshold(
    tmp_path: Path, fakes: dict[str, Any], severity: str, fail_on: str, exit_code: int
) -> None:
    fakes["near"] = [_finding(severity, count=1, check="near_dups")]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(
        app,
        ["audit", str(spec), "--out", str(tmp_path / "out"), "--fail-on", fail_on],
    )
    assert result.exit_code == exit_code, result.output


def test_audit_zero_count_high_finding_passes(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["exact"] = [_finding("high", count=0)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output


SHORTCUT_VIEWS = """checks:
  shortcut:
    views:
      - {name: headline_only, kind: text_fields, fields: [text]}
"""


def test_audit_flagged_shortcut_fails_at_high(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["shortcuts"] = [_shortcut(flagged=True)]
    spec = _spec_yaml(tmp_path / "spec.yaml", extra=SHORTCUT_VIEWS)
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 1, result.output


def test_audit_unflagged_shortcut_passes(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["shortcuts"] = [_shortcut(flagged=False)]
    spec = _spec_yaml(tmp_path / "spec.yaml", extra=SHORTCUT_VIEWS)
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output


# --- exit 2: usage and runtime errors --------------------------------------------------------


def test_unknown_fail_on_is_usage_error(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--fail-on", "critical"])
    assert result.exit_code == 2
    assert "--fail-on" in result.output


def test_missing_spec_is_exit_2(tmp_path: Path, fakes: dict[str, Any]) -> None:
    result = runner.invoke(app, ["audit", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_invalid_spec_is_exit_2(tmp_path: Path, fakes: dict[str, Any]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset:\n  name: only-a-name\n", encoding="utf-8")
    result = runner.invoke(app, ["audit", str(bad), "--out", str(tmp_path / "out")])
    assert result.exit_code == 2


def test_runtime_error_is_exit_2(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["raise"] = RuntimeError("boom")
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "RuntimeError: boom" in result.output


def test_near_threshold_out_of_range_is_exit_2(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "near", str(spec), "--threshold", "1.5"])
    assert result.exit_code == 2


def test_ngram_without_reference_is_exit_2(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "ngram", str(spec)])
    assert result.exit_code == 2
    assert "no reference corpus" in result.output


# --- specs and multi-spec runs ---------------------------------------------------------------


def test_bare_dataset_spec_is_accepted(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "gsm8k.yaml", layout="dataset")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output


def test_multiple_specs_write_one_folder_each(tmp_path: Path, fakes: dict[str, Any]) -> None:
    a = _spec_yaml(tmp_path / "a.yaml")
    b = _spec_yaml(tmp_path / "b.yaml", layout="dataset")
    out = tmp_path / "out"
    result = runner.invoke(app, ["audit", str(a), str(b), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "a" / "report.json").is_file()
    assert (out / "b" / "report.json").is_file()


def test_checks_config_is_recorded_in_env_info(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "spec.yaml", extra="checks:\n  labels: false\n")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out"), "--json"])
    assert result.exit_code == 0, result.output
    report = _audit_json(result.stdout)
    assert report.env_info["checks"]["labels"] is False


# --- --json output ---------------------------------------------------------------------------


def test_audit_json_stdout_is_a_valid_report(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["exact"] = [_finding("high", count=1)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out"), "--json"])
    assert result.exit_code == 1
    report = _audit_json(result.stdout)
    assert report.schema_version == "1"
    assert report.findings[0].severity == "high"


def test_check_exact_json_lists_findings(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["exact"] = [_finding("low", count=1)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "exact", str(spec), "--json"])
    assert result.exit_code == 0, result.output
    items = json.loads(result.stdout)
    assert isinstance(items, list)
    assert [Finding.model_validate(x).check for x in items] == ["exact_across"]


def test_check_exact_text_output_and_exit(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["exact"] = [_finding("high", count=1)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "exact", str(spec)])
    assert result.exit_code == 1
    assert "exact_across" in result.output


def test_check_near_threshold_override_is_passed(
    tmp_path: Path, fakes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, float] = {}

    def near(records: Any, text_fields: Any, config: Any, *, seed: int = 0) -> list[Finding]:
        seen["threshold"] = config.threshold
        return []

    monkeypatch.setattr(cli, "find_near_duplicates", near)
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "near", str(spec), "--threshold", "0.6", "--json"])
    assert result.exit_code == 0, result.output
    assert seen["threshold"] == pytest.approx(0.6)


def test_check_shortcut_json_lists_results(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["shortcuts"] = [_shortcut(flagged=True)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "shortcut", str(spec), "--json"])
    assert result.exit_code == 1
    items = json.loads(result.stdout)
    assert ShortcutResult.model_validate(items[0]).flagged is True


def test_check_ngram_with_reference_file(tmp_path: Path, fakes: dict[str, Any]) -> None:
    ref = tmp_path / "ref.yaml"
    ref.write_text(
        "kind: jsonl\nid_or_path: ref\nconfig: null\nrevision: null\n"
        "split_map: {reference: reference}\n",
        encoding="utf-8",
    )
    fakes["ngram"] = _finding("high", 2, check="ngram_overlap")
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "ngram", str(spec), "--reference", str(ref), "--json"])
    assert result.exit_code == 1, result.output
    items = json.loads(result.stdout)
    assert items[0]["check"] == "ngram_overlap"


# --- canary (real implementation) ------------------------------------------------------------


def test_canary_add_then_scan_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    added = runner.invoke(app, ["canary", "add", str(src), str(dst), "--json"])
    assert added.exit_code == 0, added.output
    guid = json.loads(added.stdout)["guid"]
    assert guid

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web.txt").write_text(f"some page mentions {guid}\n", encoding="utf-8")
    (corpus / "clean.txt").write_text("nothing here\n", encoding="utf-8")

    hit = runner.invoke(app, ["canary", "scan", guid, str(corpus), "--json"])
    assert hit.exit_code == 1, hit.output
    hits = json.loads(hit.stdout)
    assert hits and hits[0]["path"].endswith("web.txt")

    miss = runner.invoke(
        app, ["canary", "scan", "00000000-0000-4000-8000-000000000000", str(corpus)]
    )
    assert miss.exit_code == 0, miss.output


def test_canary_scan_missing_path_is_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["canary", "scan", "abc", str(tmp_path / "missing")])
    assert result.exit_code == 2


# --- demo ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    """Run the demo once with the real checks. Skip if a check is still not implemented."""
    out = tmp_path_factory.mktemp("demo_cli")
    result = runner.invoke(app, ["demo", "--out", str(out), "--json"])
    if result.exit_code == 2 and "NotImplementedError" in result.output:
        pytest.skip("a check still raises NotImplementedError")
    return out, result


def test_demo_runs_end_to_end_and_is_labelled_synthetic(demo_run: tuple[Path, Any]) -> None:
    out, result = demo_run
    assert result.exit_code == 1, result.output
    assert "SYNTHETIC" in result.output
    report = _audit_json(result.stdout)
    assert report.dataset.startswith("SYNTHETIC")


def test_demo_writes_manifest_and_planted_table(demo_run: tuple[Path, Any]) -> None:
    out, _ = demo_run
    manifest = DemoManifest.model_validate_json((out / "manifest.json").read_text("utf-8"))
    rows = json.loads((out / "planted_vs_found.json").read_text("utf-8"))
    planted_categories = {leak.category for leak in manifest.leaks}
    row_categories = {row["category"] for row in rows}
    assert planted_categories <= row_categories | {"near_cross_split_total"}
    statuses = {row["category"]: row["status"] for row in rows}
    assert statuses["exact_cross_split"] == "found"
    assert statuses["exact_within_train"] == "found"
    assert statuses["ngram_reference"] == "found"
    assert statuses["shortcut_token"] == "found"
    assert statuses["near_cross_split_total"] in {"found", "partial"}


def test_demo_writes_all_report_files(demo_run: tuple[Path, Any]) -> None:
    out, _ = demo_run
    for name in ("report.json", "report.md", "report.html", "audit.yaml", "manifest.json"):
        assert (out / name).is_file(), name
    assert (out / "data" / "train.jsonl").is_file()
    assert (out / "reference" / "reference.jsonl").is_file()


# --- review B1, B5 and fail-on none ----------------------------------------------------------


def test_json_error_is_one_json_object_on_stdout(tmp_path: Path) -> None:
    """With --json, an error is printed to stdout as {"error", "exit_code": 2}."""
    result = runner.invoke(app, ["audit", str(tmp_path / "missing.yaml"), "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 2
    assert "spec file not found" in payload["error"]


def test_text_error_keeps_stdout_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["audit", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_check_json_error_is_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["check", "near", str(tmp_path / "missing.yaml"), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["exit_code"] == 2


def test_fail_on_none_exits_0_even_with_a_high_finding(
    tmp_path: Path, fakes: dict[str, Any]
) -> None:
    fakes["exact"] = [_finding("high", count=2)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(
        app, ["audit", str(spec), "--out", str(tmp_path / "out"), "--fail-on", "none"]
    )
    assert result.exit_code == 0, result.output


def test_fail_on_none_is_accepted_by_check_commands(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["near"] = [_finding("high", count=2, check="near")]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["check", "near", str(spec), "--fail-on", "none"])
    assert result.exit_code == 0, result.output


def test_unknown_fail_on_value_is_a_usage_error(tmp_path: Path, fakes: dict[str, Any]) -> None:
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--fail-on", "critical", "--json"])
    assert result.exit_code == 2
    assert "none" in json.loads(result.stdout)["error"]


def test_json_report_carries_rate_intervals(tmp_path: Path, fakes: dict[str, Any]) -> None:
    fakes["exact"] = [_finding("low", count=1)]
    spec = _spec_yaml(tmp_path / "spec.yaml")
    result = runner.invoke(app, ["audit", str(spec), "--out", str(tmp_path / "out")])
    assert result.exit_code in (0, 1), result.output
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    finding = report["findings"][0]
    assert 0.0 <= finding["details"]["rate_ci_low"] <= finding["rate"]
    assert finding["rate"] <= finding["details"]["rate_ci_high"] <= 1.0
