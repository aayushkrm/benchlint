"""Tests for the hub headline: the schema subset validator, item formatting from fixture results,
the deterministic E3 selection rule, and the check that results/headline.json is current."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from benchlint.experiments import REPO_ROOT, RESULTS_DIR, write_json
from benchlint.headline import (
    LINKS,
    MAX_ITEMS,
    build_headline,
    environment_text,
    headline_text,
    load_schema,
    select_cross_findings,
)
from benchlint.jsonsubset import validate

SCRIPTS = REPO_ROOT / "scripts"


def _render_readme() -> Any:
    spec = importlib.util.spec_from_file_location("_render_for_test", SCRIPTS / "render_readme.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- validator -----------------------------------------------------------------------------


def test_validator_accepts_and_rejects_by_keyword() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a"],
        "properties": {
            "a": {"type": "string", "minLength": 1, "pattern": "^x"},
            "b": {"type": ["integer", "null"], "minimum": 0},
            "c": {"type": "array", "maxItems": 2, "items": {"const": 1}},
        },
    }
    assert validate({"a": "xy", "b": None, "c": [1]}, schema) == []
    assert validate({"a": "xy", "b": 2}, schema) == []
    assert validate({"b": 1}, schema)  # missing required key
    assert validate({"a": "xy", "extra": 1}, schema)  # additionalProperties false
    assert validate({"a": "yy"}, schema)  # pattern
    assert validate({"a": ""}, schema)  # minLength
    assert validate({"a": "xy", "b": -1}, schema)  # minimum
    assert validate({"a": "xy", "b": True}, schema)  # bool is not an integer
    assert validate({"a": "xy", "c": [1, 1, 1]}, schema)  # maxItems
    assert validate({"a": "xy", "c": [2]}, schema)  # const inside items


def test_validator_refuses_keywords_it_does_not_implement() -> None:
    with pytest.raises(ValueError):
        validate({}, {"type": "object", "oneOf": []})


def test_committed_schema_is_a_draft_2020_12_object_with_the_contract_keys() -> None:
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["required"] == [
        "schema_version",
        "tool",
        "tagline",
        "quickstart",
        "environment",
        "items",
        "links",
    ]
    assert schema["properties"]["items"]["maxItems"] == MAX_ITEMS


def test_links_match_the_contract() -> None:
    assert LINKS["repo"] == "https://github.com/aayushkrm/benchlint"
    assert LINKS["site"] == "https://aayushkrm.github.io/benchlint/"
    assert LINKS["release"] is None


# --- fixtures ------------------------------------------------------------------------------


def _report(findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "1", "dataset": "x", "findings": findings}


def _finding(check: str, scope: str, count: int, denom: int, splits: list[str]) -> dict[str, Any]:
    return {
        "check": check,
        "splits": splits,
        "count": count,
        "denominator": denom,
        "rate": count / denom if denom else 0.0,
        "details": {"scope": scope},
    }


def _fixture(root: Path, *, with_e3: bool = True) -> Path:
    """A results folder shaped like the committed one (full mode), built from fixtures."""
    write_json(
        root / "e1" / "summary.json",
        {
            "mode": "full",
            "threshold": 0.8,
            "shingle": 5,
            "lsh_recall_at_threshold": {
                "n": 10,
                "found": 9,
                "recall": 0.9,
                "ci_low": 0.6,
                "ci_high": 0.98,
                "low_n": False,
            },
            "all_near": {
                "n": 20,
                "found": 9,
                "recall": 0.45,
                "ci_low": 0.25,
                "ci_high": 0.67,
                "low_n": False,
            },
            "missed_at_or_above_threshold": 1,
            "found_below_threshold": 0,
        },
    )
    write_json(
        root / "e1" / "calibration.json",
        {
            "mode": "full",
            "settings": {"length_bins": {"short": "under 34 words"}},
            "rows": [
                {
                    "edit": "all",
                    "size": "all",
                    "length": "short",
                    "n": 40,
                    "found": 2,
                    "recall": 0.05,
                    "ci_low": 0.01,
                    "ci_high": 0.17,
                    "low_n": False,
                }
            ],
        },
    )
    write_json(
        root / "e1" / "env.json",
        {
            "cpu_model": "Test CPU",
            "cores_physical": 4,
            "cores_logical": 8,
            "ram_bytes": 17179869184,
            "os_name": "Darwin",
            "os_version": "14.8.9",
            "machine": "x86_64",
            "python_version": "3.11.15",
        },
    )
    write_json(
        root / "e2" / "snli.json",
        {
            "mode": "full",
            "dataset": "snli",
            "n_train": 100000,
            "n_eval": 9824,
            "accuracy": {"value": 0.644, "ci_low": 0.635, "ci_high": 0.654},
            "majority": 0.328,
            "literature": {"value": 0.67, "citation": "x"},
            "gap_points": -2.6,
            "flagged": True,
        },
    )
    if with_e3:
        write_json(
            root / "e3" / "imdb.json",
            _report(
                [
                    _finding("exact", "cross", 5, 100, ["train", "test"]),
                    _finding("near", "cross_split", 9, 200, ["test", "train"]),
                    _finding("near", "within_split", 50, 200, ["test", "train"]),
                ]
            ),
        )
        write_json(
            root / "e3" / "gsm8k.json",
            _report(
                [
                    _finding("exact", "cross", 0, 10, ["train", "test"]),
                    _finding("near", "cross_split", 0, 10, ["test", "train"]),
                ]
            ),
        )
        write_json(
            root / "e3" / "summary.json",
            {
                "mode": "full",
                "datasets": [
                    {"name": "imdb", "status": "ok", "report": "imdb.json"},
                    {"name": "gsm8k", "status": "ok", "report": "gsm8k.json"},
                    {"name": "snli", "status": "error", "error": "boom"},
                ],
            },
        )
    return root


# --- build_headline -------------------------------------------------------------------------------


def test_build_headline_from_fixtures_has_the_expected_items(tmp_path: Path) -> None:
    headline = build_headline(_fixture(tmp_path / "results"))
    ids = [item["id"] for item in headline["items"]]
    assert ids == [
        "e1_lsh_recall",
        "e1_short_texts",
        "e2_snli",
        "e3_imdb_cross_near",
    ]
    assert len(ids) <= MAX_ITEMS
    assert validate(headline, load_schema()) == []


def test_items_format_their_values_from_the_fixture_numbers(tmp_path: Path) -> None:
    items = {i["id"]: i for i in build_headline(_fixture(tmp_path / "r"))["items"]}
    lsh = items["e1_lsh_recall"]
    assert lsh["value"] == (
        "90.0% at true Jaccard at or above 0.8 (9 of 10); overall 45.0% (9 of 20)"
    )
    assert lsh["n"] == 10 and lsh["ci"] == "[60.0%, 98.0%]"
    assert lsh["synthetic"] is True and lsh["featured"] is True
    assert "Missed by LSH retrieval: 1" in lsh["detail"]
    short = items["e1_short_texts"]
    assert short["value"] == "5.0% (2 of 40)"
    assert "under 34 words" in short["label"]
    snli = items["e2_snli"]
    assert snli["value"] == "64.4% accuracy (literature 67.0%, reported)"
    assert snli["n"] == 9824 and snli["synthetic"] is False
    assert "Gap -2.6 points" in snli["detail"]
    assert "Gururangan et al. 2018" in snli["detail"]


def test_e3_shows_only_the_top_finding_and_it_is_featured(tmp_path: Path) -> None:
    items = {i["id"]: i for i in build_headline(_fixture(tmp_path / "r"))["items"]}
    assert items["e3_imdb_cross_near"]["featured"] is True
    assert "e3_gsm8k_cross_exact" not in items
    assert items["e3_imdb_cross_near"]["value"] == "9 of 200 items (4.50%)"
    assert "within" not in items["e3_imdb_cross_near"]["detail"].split(".")[0]


def test_empty_results_give_an_empty_valid_headline(tmp_path: Path) -> None:
    empty = tmp_path / "none"
    empty.mkdir()
    headline = build_headline(empty)
    assert headline["items"] == []
    assert headline["environment"].startswith("No environment record yet")
    assert validate(headline, load_schema()) == []


def test_environment_text_names_the_machine() -> None:
    text = environment_text(
        {
            "cpu_model": "Intel X",
            "cores_physical": 4,
            "cores_logical": 8,
            "ram_bytes": 16 * 2**30,
            "os_name": "Darwin",
            "os_version": "14.8.9",
            "machine": "x86_64",
            "python_version": "3.11.15",
        }
    )
    assert text.startswith("Intel X, 4 physical / 8 logical cores, 16 GB RAM, macOS 14.8.9 x86_64")
    assert "CPU only" in text and "Python 3.11.15" in text
    assert "/Users/" not in text


# --- E3 selection ----------------------------------------------------------------------------


def test_selection_takes_the_highest_cross_count_per_benchmark_then_the_largest() -> None:
    per_dataset = [
        (
            "a",
            [
                _finding("exact", "cross", 3, 100, ["train", "test"]),
                _finding("near", "cross_split", 8, 100, ["test", "train"]),
                _finding("near", "within_split", 99, 100, ["test", "train"]),
            ],
        ),
        ("b", [_finding("exact", "cross", 20, 100, ["train", "test"])]),
        ("c", [_finding("near", "cross_split", 1, 100, ["test", "train"])]),
        ("d", [_finding("labels", "cross", 50, 100, ["train", "test"])]),
    ]
    picked = select_cross_findings(per_dataset)
    assert [(name, f["check"], f["count"]) for name, f in picked] == [("b", "exact", 20)]


def test_selection_ties_prefer_the_higher_rate_then_exact_then_name() -> None:
    tie_rate = [
        ("z", [_finding("exact", "cross", 4, 100, ["train", "test"])]),
        ("y", [_finding("exact", "cross", 4, 50, ["train", "test"])]),
    ]
    assert [name for name, _ in select_cross_findings(tie_rate)] == ["y"]
    tie_all = [
        (
            "n",
            [
                _finding("near", "cross_split", 0, 10, ["test", "train"]),
                _finding("exact", "cross", 0, 10, ["train", "test"]),
            ],
        ),
    ]
    ((name, chosen),) = select_cross_findings(tie_all)
    assert (name, chosen["check"]) == ("n", "exact")


def test_selection_ignores_benchmarks_without_cross_findings() -> None:
    assert (
        select_cross_findings([("only_within", [_finding("near", "within_split", 5, 9, [])])]) == []
    )


# --- results/headline.json is current --------------------------------------------------------


def test_committed_headline_validates_and_points_at_result_files() -> None:
    headline = build_headline(RESULTS_DIR)
    assert validate(headline, load_schema()) == []
    for item in headline["items"]:
        assert (REPO_ROOT / item["source"]).is_file(), item["source"]


def test_render_check_fails_when_headline_is_missing_then_passes(tmp_path: Path) -> None:
    results = _fixture(tmp_path / "results")
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- results:start -->\nx\n<!-- results:end -->\n"
        "<!-- results:start -->\nx\n<!-- results:end -->\n",
        encoding="utf-8",
    )
    render = _render_readme()
    assert render.main(["--headline", "--check", "--results", str(results)]) == 1
    assert render.main(["--headline", "--results", str(results)]) == 0
    assert render.main(["--headline", "--check", "--results", str(results)]) == 0
    written = (results / "headline.json").read_text(encoding="utf-8")
    assert written == headline_text(build_headline(results))
    assert json.loads(written)["tool"] == "benchlint"


def test_e4_item_reports_the_pre_registered_count_and_the_supplementary_count(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path / "r")
    arc = {
        "n_eval": 300,
        "n_train": 2251,
        "checks": {
            "exact": {"matched": 0, "n": 300, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.012},
            "near": {"matched": 0, "n": 300, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.012},
        },
        "supplementary": {
            "n_eval": 300,
            "n_train": 2251,
            "checks": {
                "exact": {"matched": 1, "n": 300, "rate": 1 / 300, "ci_low": 0.0, "ci_high": 0.02},
                "near": {"matched": 2, "n": 300, "rate": 2 / 300, "ci_low": 0.0, "ci_high": 0.02},
            },
        },
    }
    passk = {
        "n_tasks": 24,
        "near": {"matched": 0},
        "pairwise": {"max": 0.206},
    }
    write_json(
        root / "e4" / "summary.json",
        {"mode": "full", "datasets": {"quantgate_arc_easy": arc, "passk": passk}},
    )
    items = {i["id"]: i for i in build_headline(root)["items"]}
    item = items["e4_quantgate_arc_easy_near"]
    assert item["value"].startswith("0 of 300")
    assert item["synthetic"] is False and item["featured"] is False
    assert "Supplementary, not pre-registered" in item["detail"]
    assert "Exact matches: 0" in item["detail"]
    assert "highest pairwise 5-gram Jaccard 0.21" in item["detail"]
    assert item["source"] == "results/e4/quantgate_arc_easy.json"
    assert validate(build_headline(root), load_schema()) == []
