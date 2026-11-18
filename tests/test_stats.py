"""Tests for seeded statistics: Wilson CI, ROC-AUC, bootstrap and permutation p-values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchlint.stats import (
    bootstrap_ci,
    macro_roc_auc_ovr,
    permutation_p_value,
    roc_auc,
    wilson_ci,
)


def test_wilson_zero_of_ten() -> None:
    low, high = wilson_ci(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_ten_of_ten() -> None:
    low, high = wilson_ci(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high == 1.0


def test_wilson_known_interior_value() -> None:
    low, high = wilson_ci(30, 100)
    assert low == pytest.approx(0.2189, abs=2e-3)
    assert high == pytest.approx(0.3958, abs=2e-3)


def test_wilson_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_roc_auc_perfect_inverse_and_known_value() -> None:
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    # Pairwise count: 3 of 4 positive-negative pairs are ordered correctly.
    assert roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), labels) == pytest.approx(0.75)


def test_roc_auc_all_tied_is_half() -> None:
    labels = np.array([0, 1, 0, 1, 1])
    assert roc_auc(np.zeros(5), labels) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan() -> None:
    assert math.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_roc_auc_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError):
        roc_auc(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2]))


def test_macro_ovr_with_signs_three_class() -> None:
    # Classes 0 and 2 are separated at the two ends of the score; class 1 sits in the middle
    # and cannot be separated by a single score (AUC 0.5). Flipping class 0 counts its signal.
    codes = np.array([0, 0, 1, 1, 2, 2])
    scores = codes.astype(float)
    assert macro_roc_auc_ovr(scores, codes) == pytest.approx(0.5)
    signs = np.array([-1.0, 1.0, 1.0])
    assert macro_roc_auc_ovr(scores, codes, signs) == pytest.approx((1.0 + 0.5 + 1.0) / 3.0)


def test_bootstrap_ci_covers_mean_and_is_seeded() -> None:
    values = np.arange(100, dtype=float)
    first = bootstrap_ci(np.mean, values, n_boot=300, seed=7)
    second = bootstrap_ci(np.mean, values, n_boot=300, seed=7)
    assert first == second
    assert first[0] < 49.5 < first[1]


def test_bootstrap_ci_constant_values_collapse() -> None:
    low, high = bootstrap_ci(np.mean, np.full(20, 0.3), n_boot=50, seed=1)
    assert low == pytest.approx(0.3)
    assert high == pytest.approx(0.3)


def test_bootstrap_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci(np.mean, np.array([]), n_boot=10, seed=0)


def test_permutation_p_value_plus_one_correction() -> None:
    # Null draws are all 0, so nothing reaches observed=1: p = (1 + 0) / (99 + 1).
    p = permutation_p_value(1.0, lambda rng: 0.0, n_perm=99, seed=0)
    assert p == pytest.approx(0.01)


def test_permutation_p_value_all_exceed_is_one() -> None:
    p = permutation_p_value(0.0, lambda rng: 1.0, n_perm=20, seed=0)
    assert p == pytest.approx(1.0)


def test_permutation_p_value_is_seeded() -> None:
    def null(rng: np.random.Generator) -> float:
        return float(rng.normal())

    first = permutation_p_value(1.0, null, n_perm=200, seed=3)
    second = permutation_p_value(1.0, null, n_perm=200, seed=3)
    assert first == second
