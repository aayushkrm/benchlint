"""Tests for benchlint.experiments: hashing, resume, full-only rendering and table text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchlint.experiments import (
    NO_RESULTS,
    SYNTHETIC_EDITS_NOTE,
    calibration_block,
    calibration_tables,
    config_hash,
    count_rate,
    findings_block,
    load_full,
    read_if_same_config,
    read_json,
    render_index_html,
    write_json,
)

BANNED = (
    "production-ready",
    "state-of-the-art",
    "sota",
    "blazing",
    "first",
    "novel",
    "robust",
    "enterprise",
)


def _calib(mode: str = "full") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for edit in ("substitution", "deletion", "insertion"):
        for size in (1, 3, 5, 10):
            for length in ("all", "short", "medium", "long"):
                n = 100 if length == "all" else 33
                found = 41 if length == "all" else 11
                rows.append(
                    {
                        "edit": edit,
                        "size": size,
                        "length": length,
                        "n": n,
                        "found": found,
                        "recall": found / n,
                        "ci_low": 0.3,
                        "ci_high": 0.5,
                        "median_jaccard": 0.5,
                        "low_n": n < 20,
                    }
                )
    return {
        "mode": mode,
        "rows": rows,
        "roles": {"exact": 200, "near": 1200, "clean": 600, "per_bucket": 100},
        "source": {"dataset": "fancyzhx/ag_news", "split": "test", "n_base": 2000},
        "settings": {
            "threshold": 0.8,
            "length_bins": {
                "short": "under 34 words",
                "medium": "34 to 40 words",
                "long": "41 words or more",
            },
        },
        "exact": {"n": 200, "found": 200, "recall": 1.0, "ci_low": 0.98, "ci_high": 1.0},
        "fpr": {
            "pairs": 10000,
            "flagged": 3,
            "rate": 0.0003,
            "ci_low": 0.0001,
            "ci_high": 0.0009,
        },
        "length_bins": {"short": 700, "medium": 700, "long": 600},
    }


def test_config_hash_is_stable_and_order_independent() -> None:
    assert config_hash({"a": 1, "b": [2, 3]}) == config_hash({"b": [2, 3], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_write_then_resume_only_with_matching_config(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_json(path, {"config_hash": "abc", "mode": "full"})
    assert read_if_same_config(path, "abc") is not None
    assert read_if_same_config(path, "zzz") is None
    assert read_if_same_config(tmp_path / "missing.json", "abc") is None


def test_smoke_results_are_never_loaded_for_rendering(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke.json"
    write_json(smoke, {"mode": "smoke"})
    full = tmp_path / "full.json"
    write_json(full, {"mode": "full"})
    assert load_full(smoke) is None
    assert load_full(full) == {"mode": "full"}
    assert load_full(tmp_path / "absent.json") is None


def test_no_results_text_when_nothing_is_rendered() -> None:
    assert calibration_block(None, None) == NO_RESULTS
    assert findings_block([], None, None) == NO_RESULTS


def test_calibration_block_shows_values_from_the_json_and_the_synthetic_label() -> None:
    text = calibration_block(_calib(), {"git_sha": "0123456789abcdef"})
    assert "41/100" in text
    assert "200/200" in text
    assert "0123456789" in text
    assert SYNTHETIC_EDITS_NOTE in text
    assert "under 34 words" in text
    assert "(low n)" not in text  # the fake cells all have n >= 20 except none marked


def test_low_n_cells_are_marked() -> None:
    calib = _calib()
    for row in calib["rows"]:
        if row["length"] == "long" and row["edit"] == "deletion" and row["size"] == 1:
            row["n"], row["found"], row["recall"], row["low_n"] = 4, 1, 0.25, True
    assert "1/4 = 25.0% (low n)" in calibration_block(calib, None)


def test_findings_block_reports_errors_and_reported_literature_values() -> None:
    e2 = [
        {
            "dataset": "snli",
            "n_eval": 9824,
            "n_train": 100000,
            "accuracy": {"value": 0.61, "ci_low": 0.6, "ci_high": 0.62},
            "majority": 0.34,
            "literature": {"value": 0.67},
            "gap_points": -6.0,
            "flagged": True,
        }
    ]
    e3 = {
        "datasets": [
            {
                "name": "gsm8k",
                "status": "ok",
                "findings": [
                    {
                        "check": "exact",
                        "severity": "low",
                        "count": 3,
                        "denominator": 100,
                        "rate": 0.03,
                        "ci_low": 0.01,
                        "ci_high": 0.08,
                    }
                ],
            },
            {"name": "snli", "status": "error", "error": "FileNotFoundError: x", "findings": []},
        ]
    }
    text = findings_block(e2, e3, None)
    assert "61.0%" in text
    assert "67% (reported)" in text
    assert "-6.0" in text
    assert "FileNotFoundError" in text
    assert "3 / 100" in text


def test_count_rate_shows_the_denominator() -> None:
    assert count_rate(0, 10000).startswith("0/10000 (0.0%")


def test_site_index_escapes_html_and_links_reports() -> None:
    page = render_index_html(
        "Title <b>",
        None,
        [],
        None,
        [("gsm8k", "e3/gsm8k/report.html", "ok")],
    )
    assert "Title &lt;b&gt;" in page
    assert 'href="e3/gsm8k/report.html"' in page


@pytest.mark.parametrize("word", BANNED)
def test_fixed_wording_avoids_banned_words(word: str) -> None:
    fixed = " ".join([NO_RESULTS, SYNTHETIC_EDITS_NOTE])
    assert word not in fixed.lower()


# --- E1 summary: LSH recall, true-Jaccard bins, length arithmetic ---------------------------


def _recall(points: list[tuple[str, float, bool]]) -> dict[str, Any]:
    """A fake recall stage with near planted pairs in the recall.json row layout."""
    rows = [
        [f"o{i}", f"c{i}", "near", edit, 1, "medium", jac, found]
        for i, (edit, jac, found) in enumerate(points)
    ]
    rows.append(["o99", "c99", "exact", "none", 0, "long", 1.0, True])
    return {"mode": "full", "config_hash": "h1", "planted_pairs": rows}


def test_lsh_recall_counts_only_pairs_at_or_above_the_threshold() -> None:
    from benchlint.experiments import e1_summary

    recall = _recall(
        [
            ("substitution", 0.85, True),
            ("substitution", 0.80, True),
            ("deletion", 0.82, False),  # a retrieval miss at or above the threshold
            ("insertion", 0.79, False),  # below the threshold: not counted as a miss
            ("insertion", 0.10, False),
        ]
    )
    summary = e1_summary(recall, 0.8)
    lsh = summary["lsh_recall_at_threshold"]
    assert (lsh["n"], lsh["found"]) == (3, 2)
    assert summary["missed_at_or_above_threshold"] == 1
    assert summary["found_below_threshold"] == 0
    assert summary["below_threshold"]["n"] == 2
    assert summary["near_planted"] == 5  # the exact copy is excluded


def test_true_jaccard_bins_put_the_threshold_edge_in_the_upper_bin() -> None:
    from benchlint.experiments import e1_summary

    recall = _recall(
        [("substitution", 0.8, True), ("substitution", 1.0, True), ("deletion", 0.499, False)]
    )
    bins = {b["bin"]: b for b in e1_summary(recall, 0.8)["by_true_jaccard"]}
    assert bins["[0.8, 0.9)"]["n"] == 1
    assert bins["[0.9, 1.0]"]["n"] == 1
    assert bins["[0.0, 0.5)"]["n"] == 1
    assert sum(b["n"] for b in bins.values()) == 3


def test_summary_proportions_carry_wilson_intervals() -> None:
    from benchlint.experiments import e1_summary
    from benchlint.report import wilson_interval

    recall = _recall([("substitution", 0.9, True)] * 9 + [("substitution", 0.9, False)])
    lsh = e1_summary(recall, 0.8)["lsh_recall_at_threshold"]
    assert lsh["recall"] == pytest.approx(0.9)
    assert (lsh["ci_low"], lsh["ci_high"]) == pytest.approx(wilson_interval(9, 10))


def test_length_arithmetic_matches_the_stated_formula() -> None:
    from benchlint.experiments import guaranteed_length, jaccard_lower_bound

    # S = n - 4 shingles; lower bound (S - 5k) / (S + 5k).
    assert jaccard_lower_bound(35, 1) == pytest.approx((31 - 5) / (31 + 5))
    assert jaccard_lower_bound(49, 1) == pytest.approx(0.8)
    assert jaccard_lower_bound(3, 1) == 0.0
    assert [guaranteed_length(k, 0.8) for k in (1, 3, 5, 10)] == [49, 139, 229, 454]


def test_write_e1_summary_refuses_mismatched_stage_files(tmp_path: Path) -> None:
    from benchlint.experiments import write_e1_summary

    write_json(tmp_path / "recall.json", {"mode": "full", "config_hash": "a", "planted_pairs": []})
    write_json(
        tmp_path / "calibration.json",
        {"config_hash": "b", "settings": {"threshold": 0.8, "shingle": 5}},
    )
    with pytest.raises(ValueError):
        write_e1_summary(tmp_path)


def test_write_e1_summary_round_trip(tmp_path: Path) -> None:
    from benchlint.experiments import write_e1_summary

    recall = _recall([("substitution", 0.9, True)])
    write_json(tmp_path / "recall.json", recall)
    write_json(
        tmp_path / "calibration.json",
        {"config_hash": "h1", "settings": {"threshold": 0.8, "shingle": 5}},
    )
    summary = write_e1_summary(tmp_path)
    assert read_json(tmp_path / "summary.json") == summary
    assert summary["lsh_recall_at_threshold"]["found"] == 1


def test_rendered_e1_block_labels_the_lsh_recall_and_the_bins() -> None:
    from benchlint.experiments import e1_summary

    summary = e1_summary(_recall([("substitution", 0.9, True), ("deletion", 0.5, False)]), 0.8)
    text = calibration_block(_calib(), None, summary)
    assert "detector recall conditional on true Jaccard >= 0.8 (LSH recall)" in text
    assert "E1 recall by true Jaccard bin" in text
    assert "Why short texts with word edits fall below the threshold" in text
    plain = calibration_block(_calib(), None, None)
    assert "(LSH recall)" not in plain


def test_summary_tables_use_the_threshold_from_the_summary() -> None:
    from benchlint.experiments import e1_summary

    summary = e1_summary(_recall([("substitution", 0.9, True)]), 0.7)
    titles = [t.title for t in calibration_tables(_calib(), summary)]
    assert any("true Jaccard >= 0.7" in t for t in titles)
