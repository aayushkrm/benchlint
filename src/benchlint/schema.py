"""Frozen public data types for benchlint.

Source of truth: DECISION.md section 4.3 (benchlint). Do not change these models without
an orchestrator-approved interface change. Every top-level report carries ``schema_version``.

Conventions used here:
- A field typed ``X | None`` with no default is required and nullable (the key must be present).
- Defaults are given only where DECISION.md states one, or an empty container for list fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceSpec(BaseModel):
    """Where a dataset (or reference corpus) comes from.

    ``split_map`` maps the split name used inside benchlint (key) to the source's own split
    name (value), for example ``{"validation": "validation_matched"}``.
    """

    kind: Literal["hf", "jsonl", "csv", "parquet"]
    id_or_path: str
    config: str | None
    revision: str | None
    split_map: dict[str, str]


class DatasetSpec(BaseModel):
    """One dataset to audit. Loaded from a YAML file under ``specs/``."""

    name: str
    source: SourceSpec
    text_fields: list[str]
    label_field: str | None
    id_field: str | None
    max_rows_per_split: int | None
    seed: int = 0


class View(BaseModel):
    """A partial-input view used by the shortcut check (e.g. hypothesis-only)."""

    name: str
    kind: Literal["text_fields", "length", "numeric"]
    fields: list[str]


class NearConfig(BaseModel):
    """MinHash LSH near-duplicate settings."""

    threshold: float = 0.8
    num_perm: int = 128
    shingle: int = 5


class NgramConfig(BaseModel):
    """13-gram decontamination settings: eval items vs a reference corpus."""

    n: int = 13
    reference: list[SourceSpec] = Field(default_factory=list)


class ShortcutConfig(BaseModel):
    """Partial-input baseline settings."""

    views: list[View] = Field(default_factory=list)
    margin_pts: float = 10.0
    n_boot: int = 1000
    n_perm: int = 200
    max_rows: int = 20000


class CheckConfig(BaseModel):
    """Which checks run and with what settings.

    Note: DECISION.md leaves the types of ``exact``, ``labels`` and ``canary`` unspecified.
    Booleans are used here as a placeholder; see the scaffold report for the open question.
    """

    exact: bool = True
    near: NearConfig = Field(default_factory=NearConfig)
    ngram: NgramConfig = Field(default_factory=NgramConfig)
    labels: bool = True
    shortcut: ShortcutConfig = Field(default_factory=ShortcutConfig)
    canary: bool = False


class Record(BaseModel):
    """One row of a split, after loading."""

    id: str
    split: str
    fields: dict[str, str]
    label: Any | None


class ExamplePair(BaseModel):
    """Two items flagged together. Texts are truncated to 200 characters by the producer."""

    a_id: str
    b_id: str
    a_split: str
    b_split: str
    similarity: float
    a_text: str
    b_text: str


class Finding(BaseModel):
    """One result of one check over one or more splits."""

    check: str
    severity: Literal["info", "low", "medium", "high"]
    splits: list[str]
    count: int
    denominator: int
    rate: float
    examples: list[ExamplePair] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ShortcutResult(BaseModel):
    """Outcome of one partial-input baseline on one view."""

    view: str
    metric: str
    value: float
    ci_low: float
    ci_high: float
    majority: float
    chance: float
    n_train: int
    n_test: int
    p_value: float | None
    flagged: bool


class AuditReport(BaseModel):
    """Top-level output of ``benchlint audit``."""

    schema_version: Literal["1"] = "1"
    dataset: str
    spec: DatasetSpec
    findings: list[Finding]
    shortcuts: list[ShortcutResult]
    env_info: dict[str, Any]
    runtime_s: float
