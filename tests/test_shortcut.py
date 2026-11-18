"""Tests for the partial-input shortcut check on small synthetic data (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from benchlint.checks.shortcut import run_shortcut_checks, shortcut_finding
from benchlint.schema import Record, ShortcutConfig, View

WORDS = ["cat", "dog", "tree", "river", "car", "house", "blue", "fast", "slow", "green"]
N_BOOT = 200
N_PERM = 99


def _config(*views: View, max_rows: int = 20000) -> ShortcutConfig:
    return ShortcutConfig(
        views=list(views), n_boot=N_BOOT, n_perm=N_PERM, max_rows=max_rows, margin_pts=10.0
    )


def _text_dataset(seed: int, n: int, *, shuffled: bool) -> list[Record]:
    """Hypothesis carries the marker "never" iff the latent class is 1.

    With ``shuffled`` the reported label is drawn independently of the marker, which is the
    no-signal control.
    """
    rng = np.random.default_rng(seed)
    records: list[Record] = []
    for split in ("train", "test"):
        for i in range(n):
            latent = int(rng.integers(0, 2))
            hyp_words = [str(w) for w in rng.choice(WORDS, size=4)]
            if latent == 1:
                hyp_words.append("never")
            label = int(rng.integers(0, 2)) if shuffled else latent
            records.append(
                Record(
                    id=f"{split}-{i}",
                    split=split,
                    fields={
                        "premise": " ".join(str(w) for w in rng.choice(WORDS, size=5)),
                        "hypothesis": " ".join(hyp_words),
                    },
                    label=label,
                )
            )
    return records


def _length_dataset(seed: int, n: int) -> list[Record]:
    """Label 1 texts are much longer than label 0 texts; the words carry no signal."""
    rng = np.random.default_rng(seed)
    records: list[Record] = []
    for split in ("train", "test"):
        for i in range(n):
            label = int(rng.integers(0, 2))
            length = int(rng.integers(20, 31)) if label == 1 else int(rng.integers(2, 6))
            text = " ".join(str(w) for w in rng.choice(WORDS, size=length))
            records.append(
                Record(id=f"{split}-{i}", split=split, fields={"hypothesis": text}, label=label)
            )
    return records


def _numeric_dataset(seed: int, n: int) -> list[Record]:
    """``signal`` is shifted by the label; ``noise`` is independent of it."""
    rng = np.random.default_rng(seed)
    records: list[Record] = []
    for split in ("train", "test"):
        for i in range(n):
            label = int(rng.integers(0, 2))
            signal = 2.0 * label + float(rng.normal())
            noise = float(rng.normal())
            records.append(
                Record(
                    id=f"{split}-{i}",
                    split=split,
                    fields={"signal": f"{signal:.6f}", "noise": f"{noise:.6f}"},
                    label=label,
                )
            )
    return records


HYPOTHESIS_VIEW = View(name="hypothesis_only", kind="text_fields", fields=["hypothesis"])
LENGTH_VIEW = View(name="length", kind="length", fields=["hypothesis"])
NUMERIC_VIEW = View(name="scan", kind="numeric", fields=["signal", "noise"])


def test_planted_token_shortcut_is_flagged() -> None:
    records = _text_dataset(seed=0, n=200, shuffled=False)
    results = run_shortcut_checks(
        records, _config(HYPOTHESIS_VIEW), train_split="train", test_split="test", seed=0
    )
    (result,) = results
    assert result.view == "hypothesis_only"
    assert result.metric == "accuracy"
    assert result.value > 0.95
    assert result.ci_low > result.majority + 0.10
    assert result.flagged is True
    assert result.p_value is not None and result.p_value <= 0.05
    assert result.chance == pytest.approx(0.5)


def test_shuffled_label_control_is_not_flagged() -> None:
    records = _text_dataset(seed=0, n=200, shuffled=True)
    (result,) = run_shortcut_checks(
        records, _config(HYPOTHESIS_VIEW), train_split="train", test_split="test", seed=0
    )
    assert result.flagged is False
    assert result.ci_low <= result.majority + 0.10


def test_planted_length_shortcut_is_flagged() -> None:
    records = _length_dataset(seed=1, n=200)
    (result,) = run_shortcut_checks(
        records, _config(LENGTH_VIEW), train_split="train", test_split="test", seed=1
    )
    assert result.view == "length"
    assert result.value > 0.95
    assert result.flagged is True


def test_numeric_planted_feature_flagged_and_noise_is_not() -> None:
    records = _numeric_dataset(seed=2, n=300)
    signal, noise = run_shortcut_checks(
        records, _config(NUMERIC_VIEW), train_split="train", test_split="test", seed=2
    )
    assert signal.view == "scan/signal"
    assert signal.metric == "roc_auc"
    assert signal.value > 0.8
    assert signal.flagged is True
    assert signal.majority == pytest.approx(0.5)

    assert noise.view == "scan/noise"
    assert noise.ci_low <= 0.5 <= noise.ci_high
    assert noise.flagged is False


def test_numeric_multiclass_uses_macro_ovr_metric() -> None:
    rng = np.random.default_rng(4)
    records: list[Record] = []
    for split in ("train", "test"):
        for i in range(200):
            label = int(rng.integers(0, 3))
            value = float(label) + 0.3 * float(rng.normal())
            records.append(
                Record(id=f"{split}-{i}", split=split, fields={"x": f"{value:.6f}"}, label=label)
            )
    (result,) = run_shortcut_checks(
        records,
        _config(View(name="x", kind="numeric", fields=["x"])),
        train_split="train",
        test_split="test",
        seed=4,
    )
    # Class 1 sits between the other two and has AUC 0.5; classes 0 and 2 are near 1.
    assert result.metric == "roc_auc_macro_ovr"
    assert 0.75 < result.value < 0.95
    assert result.chance == pytest.approx(0.5)


def test_multiclass_accuracy_chance_is_one_over_classes() -> None:
    rng = np.random.default_rng(5)
    records: list[Record] = []
    for split in ("train", "test"):
        for i in range(120):
            records.append(
                Record(
                    id=f"{split}-{i}",
                    split=split,
                    fields={"hypothesis": " ".join(str(w) for w in rng.choice(WORDS, size=3))},
                    label=f"class_{int(rng.integers(0, 4))}",
                )
            )
    (result,) = run_shortcut_checks(
        records, _config(HYPOTHESIS_VIEW), train_split="train", test_split="test", seed=5
    )
    assert result.chance == pytest.approx(0.25)
    assert result.flagged is False


def test_run_is_deterministic_given_seed() -> None:
    text_records = _text_dataset(seed=6, n=100, shuffled=False)
    numeric_records = _numeric_dataset(seed=6, n=150)
    text_config = _config(HYPOTHESIS_VIEW, LENGTH_VIEW)
    numeric_config = _config(NUMERIC_VIEW)
    for records, config in ((text_records, text_config), (numeric_records, numeric_config)):
        first = run_shortcut_checks(records, config, train_split="train", test_split="test", seed=9)
        second = run_shortcut_checks(
            records, config, train_split="train", test_split="test", seed=9
        )
        assert first == second


def test_max_rows_caps_each_split() -> None:
    records = _text_dataset(seed=7, n=200, shuffled=False)
    (result,) = run_shortcut_checks(
        records,
        _config(HYPOTHESIS_VIEW, max_rows=50),
        train_split="train",
        test_split="test",
        seed=7,
    )
    assert result.n_train == 50
    assert result.n_test == 50


def test_missing_field_is_rejected() -> None:
    records = _text_dataset(seed=8, n=20, shuffled=False)
    bad = View(name="bad", kind="text_fields", fields=["no_such_field"])
    with pytest.raises(ValueError, match="no_such_field"):
        run_shortcut_checks(records, _config(bad), train_split="train", test_split="test", seed=0)


def test_non_numeric_value_in_numeric_view_is_rejected() -> None:
    records = _text_dataset(seed=8, n=20, shuffled=False)
    bad = View(name="bad", kind="numeric", fields=["hypothesis"])
    with pytest.raises(ValueError, match="not numeric"):
        run_shortcut_checks(records, _config(bad), train_split="train", test_split="test", seed=0)


def test_shortcut_finding_aggregates_flags() -> None:
    records = _text_dataset(seed=0, n=200, shuffled=False)
    results = run_shortcut_checks(
        records, _config(HYPOTHESIS_VIEW), train_split="train", test_split="test", seed=0
    )
    finding = shortcut_finding(results, ["train", "test"])
    assert finding.check == "shortcut"
    assert finding.severity == "high"
    assert (finding.count, finding.denominator, finding.rate) == (1, 1, 1.0)
    assert finding.details["flagged_views"] == ["hypothesis_only"]


def test_shortcut_finding_without_flags_is_info() -> None:
    records = _text_dataset(seed=0, n=200, shuffled=True)
    results = run_shortcut_checks(
        records, _config(HYPOTHESIS_VIEW), train_split="train", test_split="test", seed=0
    )
    finding = shortcut_finding(results, ["train", "test"])
    assert finding.severity == "info"
    assert finding.count == 0
