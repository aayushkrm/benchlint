"""Statistics for the shortcut check: seeded bootstrap CIs, permutation p-values, ROC-AUC.

All functions are deterministic given their explicit ``seed`` argument and use numpy only
(no global random state, no sklearn).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist

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

    ``values`` holds one observation per row (axis 0). ``statistic`` receives a resampled array
    with the same number of rows and returns a float. Resamples where the statistic is NaN
    (for example a ROC-AUC with a single class) are dropped. Returns ``(ci_low, ci_high)`` at
    level ``1 - alpha``.
    """
    arr = np.asarray(values)
    if arr.ndim == 0 or arr.shape[0] == 0:
        raise ValueError("bootstrap_ci needs at least one row")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = int(arr.shape[0])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = statistic(arr[idx])
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        raise ValueError("statistic is undefined on every bootstrap resample")
    low, high = np.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def permutation_p_value(
    observed: float,
    null_statistic: Callable[[np.random.Generator], float],
    *,
    n_perm: int,
    seed: int,
) -> float:
    """One-sided permutation p-value: share of null draws >= ``observed`` (with +1 correction).

    ``null_statistic`` receives a seeded generator and returns the statistic computed under one
    random permutation (for example with the labels shuffled). The p-value is
    ``(1 + #{null >= observed}) / (n_perm + 1)``, so it is never zero.
    """
    if math.isnan(observed):
        raise ValueError("observed statistic is NaN")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")
    rng = np.random.default_rng(seed)
    null = np.array([null_statistic(rng) for _ in range(n_perm)], dtype=float)
    exceed = int(np.sum(null >= observed - 1e-12))
    return (1.0 + exceed) / (n_perm + 1.0)


def wilson_ci(k: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` trials.

    Returns ``(low, high)`` clipped to [0, 1]; the endpoints are exactly 0 when ``k == 0`` and
    exactly 1 when ``k == n``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2 = z * z
    p = k / n
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of a numeric score against binary ``labels`` (ties count as half).

    ``labels`` must be boolean or coded 0/1; ``True``/``1`` is the positive class. Computed as
    the normalised Mann-Whitney U statistic with average ranks for ties. Returns NaN when the
    labels contain only one class, since the AUC is undefined then.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be boolean or coded 0/1")
    pos = y.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    u = float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def macro_roc_auc_ovr(
    scores: np.ndarray, codes: np.ndarray, signs: np.ndarray | None = None
) -> float:
    """Macro-averaged one-vs-rest ROC-AUC of one score against integer class ``codes``.

    ``codes`` are non-negative integers 0..K-1. Class ``k`` is scored against all other classes
    using ``signs[k] * scores`` (``signs`` holds +1 or -1 per class, default all +1), so a class
    the score ranks in the opposite direction still counts as signal. Classes whose AUC is
    undefined are skipped. Returns NaN when no class has a defined AUC.
    """
    s = np.asarray(scores, dtype=float).ravel()
    c = np.asarray(codes).ravel()
    if s.shape != c.shape:
        raise ValueError("scores and codes must have the same length")
    if signs is not None and np.asarray(signs).size < int(c.max(initial=0)) + 1:
        raise ValueError("signs must have one entry per class code")
    aucs: list[float] = []
    for k in np.unique(c):
        sign = 1.0 if signs is None else float(np.asarray(signs)[int(k)])
        aucs.append(roc_auc(sign * s, c == k))
    defined = [a for a in aucs if not math.isnan(a)]
    if not defined:
        return float("nan")
    return float(np.mean(defined))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks of ``values``, with tied values sharing their average rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts).astype(float)
    average = cumulative - (counts - 1) / 2.0
    return np.asarray(average[np.asarray(inverse).ravel()], dtype=float)
