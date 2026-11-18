"""Partial-input baselines: does a single view of the input predict the label?

Views: TF-IDF (1-2 grams) + logistic regression on one field, a length-only view, and a
single numeric feature scanned by ROC-AUC. Each result is compared with majority and chance,
with a seeded bootstrap CI and a permutation p-value on a subsample of at most
``config.max_rows`` rows. Flagged when the lower CI bound exceeds majority + ``margin_pts``.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchlint.schema import Record, ShortcutConfig, ShortcutResult


def run_shortcut_checks(
    records: Sequence[Record],
    config: ShortcutConfig,
    *,
    train_split: str,
    test_split: str,
    seed: int = 0,
) -> list[ShortcutResult]:
    """Fit each view on ``train_split`` and score it on ``test_split``."""
    raise NotImplementedError
