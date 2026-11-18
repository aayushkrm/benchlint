"""Statistics for the shortcut check: seeded bootstrap CIs, permutation p-values, ROC-AUC.

All functions are deterministic given ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over resampled rows of ``values``.

    Returns ``(ci_low, ci_high)`` at level ``1 - alpha``.
    """
    raise NotImplementedError


def permutation_p_value(
    observed: float,
    null_statistic: Callable[[np.random.Generator], float],
    *,
    n_perm: int,
    seed: int,
) -> float:
    """One-sided permutation p-value: share of null draws >= ``observed`` (with +1 correction)."""
    raise NotImplementedError


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of a single numeric feature against binary ``labels`` (ties count as half)."""
    raise NotImplementedError
