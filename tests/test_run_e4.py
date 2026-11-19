"""Offline tests for scripts/run_e4.py: provenance, text definitions, matching, the pairwise
summary and passk loading. Nothing here downloads a dataset or reads a real sibling checkout."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from benchlint.experiments import REPO_ROOT
from benchlint.schema import NearConfig, Record

SCRIPT = REPO_ROOT / "scripts" / "run_e4.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_run_e4_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e4 = _load()
TEXT = "the quick brown fox jumps over the lazy dog near the quiet river bank today"


def _rec(split: str, key: str, text: str) -> Record:
    return Record(id=f"{split}:{key}", split=split, fields={"text": text}, label=None)


def test_arc_text_is_question_then_choices_in_order() -> None:
    fields = {"question": "Why?", "choices": json.dumps({"text": ["b", "a"], "label": ["1", "2"]})}
    assert e4.arc_text(fields) == "Why? b a"
    assert e4.arc_question_text(fields) == "Why?"


def test_gsm8k_texts_match_quantgate_and_the_e3_spec() -> None:
    fields = {"question": "How many?", "answer": "3 #### 3"}
    assert e4.gsm8k_text(fields) == "How many?"
    assert e4.gsm8k_question_and_answer(fields) == "How many? 3 #### 3"


def test_eval_records_refuse_a_missing_id() -> None:
    by_key = {"1": _rec("test", "1", TEXT)}
    text_of = lambda fields: fields["text"]  # noqa: E731
    assert [r.id for r in e4.eval_records_from(by_key, text_of, ["1"])] == ["test:1"]
    with pytest.raises(e4.E4Error):
        e4.eval_records_from(by_key, text_of, ["1", "missing"])


def test_eval_matches_count_only_cross_split_partners() -> None:
    train = [
        _rec("train", "a", TEXT),
        _rec("train", "b", "completely different words about rivers"),
    ]
    evals = [
        Record(id="test:x", split="test", fields={"text": TEXT}, label=None),
        Record(
            id="test:y",
            split="test",
            fields={"text": "an unrelated question about planets"},
            label=None,
        ),
    ]
    result = e4.eval_matches([*train, *evals], {"test:x", "test:y"}, NearConfig())
    assert result["exact"]["matched"] == 1
    assert result["exact"]["n"] == 2
    assert result["near"]["matched"] == 1
    assert result["near"]["ci_low"] <= 0.5 <= result["near"]["ci_high"]
    assert len(result["exact"]["examples"]) == 1
    assert len(result["exact"]["examples"][0]["a_text"]) <= 200


def test_eval_matches_ignore_train_only_duplicates() -> None:
    """Two identical train items are not an eval match: only eval-to-train pairs count."""
    train = [_rec("train", "a", TEXT), _rec("train", "b", TEXT)]
    evals = [Record(id="test:x", split="test", fields={"text": "unrelated words here"}, label=None)]
    result = e4.eval_matches([*train, *evals], {"test:x"}, NearConfig())
    assert result["exact"]["matched"] == 0
    assert result["near"]["matched"] == 0


def test_pairwise_summary_counts_every_pair_and_orders_the_top_pairs() -> None:
    records = [
        _rec("tasks", "t1", "Order 10 g of compound C01 for a QC run."),
        _rec("tasks", "t2", "Order 10 g of compound C02 for a QC run."),
        _rec("tasks", "t3", "Request approval for a totally different action now."),
    ]
    summary = e4.pairwise_summary(records)
    assert summary["pairs"] == 3
    assert summary["min"] <= summary["median"] <= summary["max"]
    assert sum(summary["histogram"]["counts"]) == 3
    assert summary["at_or_above_threshold"]["n"] == 3
    jaccards = [t["jaccard"] for t in summary["top_pairs"]]
    assert jaccards == sorted(jaccards, reverse=True)
    assert summary["top_pairs"][0]["a_id"] == "tasks:t1"


def test_groups_join_identical_items() -> None:
    class Finding:
        details = {"pairs": [["a", "b", 1.0], ["b", "c", 1.0], ["x", "y", 1.0]]}

    assert e4._groups(Finding()) == [["a", "b", "c"], ["x", "y"]]


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def test_sibling_provenance_records_relative_paths_only(tmp_path: Path) -> None:
    repo = tmp_path / "sibling"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "a.txt").write_text("x", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    clean = e4.sibling_provenance("sib", repo)
    assert clean["git_dirty"] is False and len(clean["git_sha"]) == 40
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    dirty = e4.sibling_provenance("sib", repo)
    assert dirty["git_dirty"] is True
    assert dirty["git_dirty_paths"] == ["a.txt"]
    assert str(tmp_path) not in json.dumps(dirty)


def test_sibling_provenance_names_a_missing_checkout_without_its_path(tmp_path: Path) -> None:
    with pytest.raises(e4.E4Error) as info:
        e4.sibling_provenance("quantgate", tmp_path / "nope")
    assert "quantgate" in str(info.value)
    assert str(tmp_path) not in str(info.value)


def test_load_instructions_reads_the_instruction_field(tmp_path: Path) -> None:
    folder = tmp_path / e4.PASSK_TASKS
    folder.mkdir(parents=True)
    (folder / "t1_01.yaml").write_text(
        "id: t1_01\ninstruction: Place an order.\n", encoding="utf-8"
    )
    (folder / "t1_02.yaml").write_text(
        "id: t1_02\ninstruction: Cancel an order.\n", encoding="utf-8"
    )
    tasks, files = e4.load_instructions(tmp_path)
    assert [t["instruction"] for t in tasks] == ["Place an order.", "Cancel an order."]
    assert [f["path"] for f in files] == [
        f"{e4.PASSK_TASKS}/t1_01.yaml",
        f"{e4.PASSK_TASKS}/t1_02.yaml",
    ]
    assert all(len(f["sha256"]) == 64 for f in files)


def test_load_instructions_refuses_a_task_without_instruction(tmp_path: Path) -> None:
    folder = tmp_path / e4.PASSK_TASKS
    folder.mkdir(parents=True)
    (folder / "bad.yaml").write_text("id: bad\n", encoding="utf-8")
    with pytest.raises(e4.E4Error):
        e4.load_instructions(tmp_path)


def test_committed_e4_results_hold_no_absolute_path_or_username() -> None:
    folder = REPO_ROOT / "results" / "e4"
    if not folder.is_dir():
        pytest.skip("E4 has not been run")
    username = Path.home().name
    for path in folder.glob("*.json"):
        text: Any = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path.name
        assert str(REPO_ROOT.parent) not in text, path.name
        assert username and username not in text, path.name
