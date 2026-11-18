"""Partial-input baselines: does a single view of the input predict the label?

Views: TF-IDF (1-2 grams) + logistic regression on the selected field(s), a length-only view
(character and word count, shallow decision tree), and a single numeric feature scanned by
ROC-AUC (one result per field, orientation fixed on the train split).

Classifier views report accuracy against the train-majority baseline and chance (1/#classes).
Numeric views report ROC-AUC against 0.5 (a constant score). Each result has a seeded bootstrap
CI on the test split and a seeded label-permutation p-value. A view is flagged when the lower CI
bound exceeds majority + ``margin_pts`` / 100. Splits are subsampled to at most ``max_rows``
rows each. Unadjusted p-values: a numeric scan over many fields is not corrected for multiple
comparisons.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from benchlint.schema import Finding, Record, ShortcutConfig, ShortcutResult, View
from benchlint.stats import bootstrap_ci, macro_roc_auc_ovr, permutation_p_value, roc_auc

_NUMERIC_MAJORITY = 0.5  # ROC-AUC of a constant score
_MAX_DEPTH_LENGTH = 3


def run_shortcut_checks(
    records: Sequence[Record],
    config: ShortcutConfig,
    *,
    train_split: str,
    test_split: str,
    seed: int = 0,
) -> list[ShortcutResult]:
    """Fit each view on ``train_split`` and score it on ``test_split``.

    Each split is subsampled (seeded) to at most ``config.max_rows`` rows. Returns one
    ``ShortcutResult`` per classifier view and one per field of each numeric view (the view
    name is then ``"<view>/<field>"``). Raises ``ValueError`` on an empty split, a missing
    label, a missing field or a non-numeric value in a numeric view.
    """
    if train_split == test_split:
        raise ValueError("train_split and test_split must differ")
    train = [r for r in records if r.split == train_split]
    test = [r for r in records if r.split == test_split]
    if not train or not test:
        raise ValueError(
            f"no rows in split {train_split!r} ({len(train)}) or {test_split!r} ({len(test)})"
        )
    rng = np.random.default_rng(seed)
    train = _cap(train, config.max_rows, rng)
    test = _cap(test, config.max_rows, rng)
    y_train, y_test, n_classes = _encode_labels(train, test)

    results: list[ShortcutResult] = []
    for view in config.views:
        _check_fields(view, [*train, *test])
        if view.kind == "text_fields":
            x_train = _joined(train, view.fields)
            x_test = _joined(test, view.fields)
            results.append(
                _text_view(view.name, x_train, x_test, y_train, y_test, n_classes, config, seed)
            )
        elif view.kind == "length":
            feat_train = _length_features(_joined(train, view.fields))
            feat_test = _length_features(_joined(test, view.fields))
            results.append(
                _length_view(
                    view.name, feat_train, feat_test, y_train, y_test, n_classes, config, seed
                )
            )
        else:
            results.extend(
                _numeric_view(view, train, test, y_train, y_test, n_classes, config, seed)
            )
    return results


def shortcut_finding(results: Sequence[ShortcutResult], splits: Sequence[str]) -> Finding:
    """Aggregate shortcut results into one ``Finding(check="shortcut")``.

    ``count`` is the number of flagged views, ``denominator`` the number of views evaluated and
    ``rate`` their ratio (0.0 when there are no views). Severity is ``"high"`` if any view is
    flagged, otherwise ``"info"``. ``details`` lists the flagged view names and every result.
    """
    flagged = [r.view for r in results if r.flagged]
    denominator = len(results)
    count = len(flagged)
    return Finding(
        check="shortcut",
        severity="high" if count else "info",
        splits=list(splits),
        count=count,
        denominator=denominator,
        rate=count / denominator if denominator else 0.0,
        details={
            "flagged_views": flagged,
            "views": [r.model_dump(mode="json") for r in results],
        },
    )


def _cap(rows: list[Record], max_rows: int, rng: np.random.Generator) -> list[Record]:
    """Seeded subsample of at most ``max_rows`` rows, kept in original order."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if len(rows) <= max_rows:
        return rows
    idx = np.sort(rng.choice(len(rows), size=max_rows, replace=False))
    return [rows[int(i)] for i in idx]


def _check_fields(view: View, rows: Sequence[Record]) -> None:
    """Fail early when a view names no fields or a field missing from some row."""
    if not view.fields:
        raise ValueError(f"view {view.name!r} lists no fields")
    for field in view.fields:
        if any(field not in r.fields for r in rows):
            raise ValueError(f"view {view.name!r} uses field {field!r}, missing from some rows")


def _labels(rows: Sequence[Record]) -> list[Any]:
    """Raw labels of ``rows``; a row without a label is an error."""
    out: list[Any] = []
    for r in rows:
        if r.label is None:
            raise ValueError(f"record {r.id!r} in split {r.split!r} has no label")
        out.append(r.label)
    return out


def _encode_labels(
    train: Sequence[Record], test: Sequence[Record]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Integer codes for train and test labels over the union of classes.

    Classes are sorted naturally (by ``str`` if the labels are not mutually comparable). For
    binary labels the positive class is code 1, the class that sorts last.
    """
    raw_train = _labels(train)
    raw_test = _labels(test)
    uniq = set(raw_train) | set(raw_test)
    try:
        classes = sorted(uniq)
    except TypeError:
        classes = sorted(uniq, key=str)
    index = {c: i for i, c in enumerate(classes)}
    y_train = np.array([index[v] for v in raw_train], dtype=int)
    y_test = np.array([index[v] for v in raw_test], dtype=int)
    return y_train, y_test, len(classes)


def _joined(rows: Sequence[Record], fields: Sequence[str]) -> list[str]:
    """Selected fields of each row joined with a space."""
    return [" ".join(r.fields[f] for f in fields) for r in rows]


def _length_features(texts: Sequence[str]) -> np.ndarray:
    """Log character and word counts, one row per text."""
    chars = np.array([len(t) for t in texts], dtype=float)
    words = np.array([len(t.split()) for t in texts], dtype=float)
    return np.column_stack([np.log1p(chars), np.log1p(words)])


def _majority_code(y_train: np.ndarray, n_classes: int) -> int:
    """Most frequent train label; ties go to the lowest code."""
    return int(np.argmax(np.bincount(y_train, minlength=n_classes)))


def _fit_predict(
    model: LogisticRegression | DecisionTreeClassifier,
    x_train: Any,
    y_train: np.ndarray,
    x_test: Any,
    n_test: int,
) -> np.ndarray:
    """Fit ``model`` and predict codes for the test rows.

    A train split with a single class yields a constant prediction (that class).
    """
    if np.unique(y_train).size < 2:
        return np.full(n_test, int(y_train[0]), dtype=int)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=int)


def _text_view(
    name: str,
    x_train: list[str],
    x_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    config: ShortcutConfig,
    seed: int,
) -> ShortcutResult:
    """TF-IDF (1-2 grams) + logistic regression on the selected text."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features_train = vectorizer.fit_transform(x_train)
    features_test = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    pred = _fit_predict(model, features_train, y_train, features_test, len(y_test))
    return _accuracy_result(name, pred, y_train, y_test, n_classes, config, seed)


def _length_view(
    name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    config: ShortcutConfig,
    seed: int,
) -> ShortcutResult:
    """Character and word count only, through a shallow decision tree."""
    model = DecisionTreeClassifier(max_depth=_MAX_DEPTH_LENGTH, random_state=seed)
    pred = _fit_predict(model, x_train, y_train, x_test, len(y_test))
    return _accuracy_result(name, pred, y_train, y_test, n_classes, config, seed)


def _accuracy_result(
    name: str,
    pred: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    config: ShortcutConfig,
    seed: int,
) -> ShortcutResult:
    """Accuracy with CI, permutation p-value, majority and chance baselines."""
    correct = (pred == y_test).astype(float)
    value = float(np.mean(correct))
    ci_low, ci_high = bootstrap_ci(_mean, correct, n_boot=config.n_boot, seed=seed)
    p_value: float | None = None
    if config.n_perm > 0:
        p_value = permutation_p_value(
            value,
            lambda rng: float(np.mean(pred == rng.permutation(y_test))),
            n_perm=config.n_perm,
            seed=seed,
        )
    majority = float(np.mean(y_test == _majority_code(y_train, n_classes)))
    return ShortcutResult(
        view=name,
        metric="accuracy",
        value=value,
        ci_low=ci_low,
        ci_high=ci_high,
        majority=majority,
        chance=1.0 / n_classes,
        n_train=int(y_train.size),
        n_test=int(y_test.size),
        p_value=p_value,
        flagged=ci_low > majority + config.margin_pts / 100.0,
    )


def _numeric_view(
    view: View,
    train: Sequence[Record],
    test: Sequence[Record],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    config: ShortcutConfig,
    seed: int,
) -> list[ShortcutResult]:
    """One ROC-AUC result per field of a numeric view."""
    if n_classes < 2:
        raise ValueError(f"numeric view {view.name!r} needs at least two classes")
    if np.unique(y_test).size < 2:
        raise ValueError(f"numeric view {view.name!r}: test split has a single class")
    out: list[ShortcutResult] = []
    for field in view.fields:
        x_train = _floats(train, field)
        x_test = _floats(test, field)
        # Orientation comes from the train split only, so the test split is not used to choose.
        signs = _orientations(x_train, y_train, n_classes)
        out.append(
            _numeric_result(
                f"{view.name}/{field}",
                x_test,
                y_test,
                n_classes,
                signs,
                config,
                seed,
                n_train=int(y_train.size),
            )
        )
    return out


def _orientations(scores_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class sign (+1 or -1) of a score, chosen on the train split.

    A class whose train ROC-AUC is below 0.5 is scored with the sign flipped, so an inverse
    relation counts as signal. Binary labels only use the entry for code 1.
    """
    signs = np.ones(n_classes, dtype=float)
    for k in range(n_classes):
        auc = roc_auc(scores_train, y_train == k)
        if not math.isnan(auc) and auc < 0.5:
            signs[k] = -1.0
    return signs


def _numeric_result(
    name: str,
    scores: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    signs: np.ndarray,
    config: ShortcutConfig,
    seed: int,
    *,
    n_train: int,
) -> ShortcutResult:
    """ROC-AUC of one score on the test split, with CI and permutation p-value."""
    value = _auc(scores, y_test, n_classes, signs)
    values = np.column_stack([scores, y_test.astype(float)])
    ci_low, ci_high = bootstrap_ci(
        lambda v: _auc(v[:, 0], v[:, 1].astype(int), n_classes, signs),
        values,
        n_boot=config.n_boot,
        seed=seed,
    )
    p_value: float | None = None
    if config.n_perm > 0:
        p_value = permutation_p_value(
            value,
            lambda rng: _auc(scores, rng.permutation(y_test), n_classes, signs),
            n_perm=config.n_perm,
            seed=seed,
        )
    return ShortcutResult(
        view=name,
        metric="roc_auc" if n_classes == 2 else "roc_auc_macro_ovr",
        value=value,
        ci_low=ci_low,
        ci_high=ci_high,
        majority=_NUMERIC_MAJORITY,
        chance=_NUMERIC_MAJORITY,
        n_train=n_train,
        n_test=int(y_test.size),
        p_value=p_value,
        flagged=ci_low > _NUMERIC_MAJORITY + config.margin_pts / 100.0,
    )


def _auc(scores: np.ndarray, codes: np.ndarray, n_classes: int, signs: np.ndarray) -> float:
    """ROC-AUC with per-class orientation: binary on code 1, macro one-vs-rest otherwise."""
    if n_classes == 2:
        return roc_auc(signs[1] * scores, codes == 1)
    return macro_roc_auc_ovr(scores, codes, signs)


def _floats(rows: Sequence[Record], field: str) -> np.ndarray:
    """A numeric field as floats; non-numeric or non-finite values are an error."""
    values: list[float] = []
    for r in rows:
        raw = r.fields[field]
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"field {field!r} is not numeric: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"field {field!r} has a non-finite value: {raw!r}")
        values.append(value)
    return np.asarray(values, dtype=float)


def _mean(values: np.ndarray) -> float:
    """Mean as a float (bootstrap statistic for accuracy)."""
    return float(np.mean(values))
