"""Public data types for benchlint: dataset specs, checks and reports.

Changes to these models need a ``schema_version`` decision and an entry in docs/deviations.md.
Every top-level report carries ``schema_version``.

Conventions used here:
- A field typed ``X | None`` with no default is required and nullable (the key must be present).
- Defaults are given only where docs/prereg.md or docs/deviations.md sets one, or an empty
  container for list fields.
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
    drop_label_values: list[Any] = Field(default_factory=list)
    """Rows whose label (compared as text) is one of these are removed before sampling.
    Used for SNLI, where ``-1`` marks an item with no gold label. Empty means keep all rows."""


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
    lsh_threshold: float = 0.7
    """LSH retrieval threshold. The effective value is ``min(threshold, lsh_threshold)``.
    Retrieving below the reporting threshold raises LSH recall; reported pairs are still
    verified by exact Jaccard at ``threshold``."""


class NgramConfig(BaseModel):
    """13-gram decontamination settings: eval items vs a reference corpus."""

    n: int = 13
    reference: list[SourceSpec] = Field(default_factory=list)
    reference_fields: list[str] = Field(default_factory=lambda: ["text"])
    """Fields of each reference row that are joined (space-separated) into its text."""


class ShortcutConfig(BaseModel):
    """Partial-input baseline settings."""

    views: list[View] = Field(default_factory=list)
    margin_pts: float = 10.0
    n_boot: int = 1000
    n_perm: int = 200
    max_rows: int = 20000


class CheckConfig(BaseModel):
    """Which checks run and with what settings.

    Note: the design docs do not specify the types of ``exact``, ``labels`` and ``canary``;
    booleans are used here.
    """

    exact: bool = True
    near: NearConfig = Field(default_factory=NearConfig)
    ngram: NgramConfig = Field(default_factory=NgramConfig)
    labels: bool = True
    shortcut: ShortcutConfig = Field(default_factory=ShortcutConfig)
    canary: bool = False


class AuditSpec(BaseModel):
    """Top-level audit file: a dataset plus the checks to run on it.

    This is what ``benchlint audit spec.yaml`` loads. A DatasetSpec alone has no place for check
    settings, so the audit file wraps it. YAML layout: ``dataset:`` (a DatasetSpec) and optional
    ``checks:``.
    """

    dataset: DatasetSpec
    checks: CheckConfig = Field(default_factory=CheckConfig)


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
