"""Near-duplicate detection with MinHash LSH (datasketch).

Word shingles of size ``config.shingle``, ``config.num_perm`` permutations and a Jaccard
threshold of ``config.threshold``. Candidate pairs from LSH are verified with exact Jaccard.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchlint.schema import Finding, NearConfig, Record


def find_near_duplicates(
    records: Sequence[Record],
    text_fields: Sequence[str],
    config: NearConfig,
    *,
    seed: int = 0,
) -> list[Finding]:
    """Return near-duplicate findings, with the Jaccard distribution in ``details``.

    The distribution is reported, not only threshold counts, to expose template-driven
    false positives (for example GSM8K phrasing).
    """
    raise NotImplementedError
