"""Exact duplicate detection: within each split and across splits.

Key is the blake2b digest of the normalised, concatenated text fields.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchlint.schema import Finding, Record


def find_exact_duplicates(records: Sequence[Record], text_fields: Sequence[str]) -> list[Finding]:
    """Return one ``Finding`` per within-split and per cross-split duplicate group set.

    Each finding lists its ``splits``, the duplicate count, the denominator (records in the
    checked splits) and up to 5 example pairs.
    """
    raise NotImplementedError
