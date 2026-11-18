"""Label conflicts: duplicate or near-duplicate items that carry different labels."""

from __future__ import annotations

from collections.abc import Sequence

from benchlint.schema import Finding, NearConfig, Record


def find_label_conflicts(
    records: Sequence[Record],
    text_fields: Sequence[str],
    *,
    near: NearConfig | None = None,
) -> list[Finding]:
    """Find groups of duplicates with more than one distinct label.

    If ``near`` is None only exact duplicates are grouped; otherwise near-duplicates too.
    """
    raise NotImplementedError
