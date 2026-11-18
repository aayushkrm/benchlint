"""JSON round-trip tests for every frozen benchlint model (DECISION.md section 4.3)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from benchlint.schema import (
    AuditReport,
    CheckConfig,
    DatasetSpec,
    ExamplePair,
    Finding,
    NearConfig,
    NgramConfig,
    Record,
    ShortcutConfig,
    ShortcutResult,
    SourceSpec,
    View,
)


def _source() -> SourceSpec:
    return SourceSpec(
        kind="hf",
        id_or_path="openai/gsm8k",
        config="main",
        revision=None,
        split_map={"train": "train", "test": "test"},
    )


def _spec() -> DatasetSpec:
    return DatasetSpec(
        name="gsm8k",
        source=_source(),
        text_fields=["question", "answer"],
        label_field=None,
        id_field=None,
        max_rows_per_split=None,
        seed=0,
    )


def _example() -> ExamplePair:
    return ExamplePair(
        a_id="0",
        b_id="7",
        a_split="train",
        b_split="test",
        similarity=0.93,
        a_text="a short text",
        b_text="a short text, slightly edited",
    )


def _finding() -> Finding:
    return Finding(
        check="exact_across",
        severity="high",
        splits=["train", "test"],
        count=2,
        denominator=100,
        rate=0.02,
        examples=[_example()],
        details={"groups": 2},
    )


def _shortcut() -> ShortcutResult:
    return ShortcutResult(
        view="hypothesis_only",
        metric="accuracy",
        value=0.67,
        ci_low=0.65,
        ci_high=0.69,
        majority=0.34,
        chance=0.33,
        n_train=1000,
        n_test=500,
        p_value=0.001,
        flagged=True,
    )


def _samples() -> list[BaseModel]:
    return [
        _source(),
        _spec(),
        View(name="length", kind="length", fields=["question"]),
        NearConfig(),
        NgramConfig(n=13, reference=[_source()]),
        ShortcutConfig(
            views=[View(name="hyp", kind="text_fields", fields=["hypothesis"])],
            margin_pts=10.0,
            n_boot=1000,
            n_perm=200,
            max_rows=20000,
        ),
        CheckConfig(),
        Record(id="0", split="train", fields={"text": "hello"}, label=1),
        Record(id="1", split="test", fields={"text": "bye"}, label=None),
        _example(),
        _finding(),
        _shortcut(),
        AuditReport(
            dataset="gsm8k",
            spec=_spec(),
            findings=[_finding()],
            shortcuts=[_shortcut()],
            env_info={"git_sha": "unknown", "python": "3.11.0"},
            runtime_s=1.5,
        ),
    ]


@pytest.mark.parametrize("model", _samples(), ids=lambda m: type(m).__name__)
def test_json_round_trip(model: BaseModel) -> None:
    text = model.model_dump_json()
    restored = type(model).model_validate_json(text)
    assert restored == model
    assert restored.model_dump_json() == text


def test_audit_report_schema_version_defaults_to_1() -> None:
    report = AuditReport(
        dataset="d",
        spec=_spec(),
        findings=[],
        shortcuts=[],
        env_info={},
        runtime_s=0.0,
    )
    assert report.schema_version == "1"
    assert '"schema_version":"1"' in report.model_dump_json()


def test_audit_report_rejects_other_schema_versions() -> None:
    with pytest.raises(ValidationError):
        AuditReport(
            schema_version="2",  # type: ignore[arg-type]
            dataset="d",
            spec=_spec(),
            findings=[],
            shortcuts=[],
            env_info={},
            runtime_s=0.0,
        )


def test_finding_rejects_unknown_severity() -> None:
    with pytest.raises(ValidationError):
        Finding(
            check="x",
            severity="critical",  # type: ignore[arg-type]
            splits=[],
            count=0,
            denominator=0,
            rate=0.0,
        )
