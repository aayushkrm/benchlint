"""Tests for benchlint.experiments: hashing, resume, full-only rendering and table text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchlint.experiments import (
    NO_RESULTS,
    SYNTHETIC_EDITS_NOTE,
    calibration_block,
    config_hash,
    count_rate,
    findings_block,
    load_full,
    read_if_same_config,
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
