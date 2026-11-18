"""Tests for the experiment and rendering scripts: the offline parts only (no network).

The scripts are loaded from scripts/ by path. Network loads (AG News, SNLI, MNLI and the
rest) are not run here; the smoke commands in docs/prereg.md and the README cover them.
"""

from __future__ import annotations

import importlib.util
import random
import string
from pathlib import Path
from types import ModuleType

import pytest

from benchlint.cli import load_audit_spec
from benchlint.experiments import NO_RESULTS, REPO_ROOT, RESULTS_DIR

SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibrate = _load("calibrate")
render_readme = _load("render_readme")
build_site = _load("build_site")
run_e3 = _load("run_e3")


def _words(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 7))) for _ in range(n)]


# --- E1 planting ----------------------------------------------------------------------------


@pytest.mark.parametrize("edit", ["substitution", "deletion", "insertion"])
@pytest.mark.parametrize("size", [1, 3, 5, 10])
def test_apply_edit_changes_exactly_size_tokens(edit: str, size: int) -> None:
    tokens = _words(40, seed=size)
    vocab = sorted(set(_words(500, seed=99)))
    out = calibrate.apply_edit(tokens, edit, size, random.Random(0), vocab)
    if edit == "substitution":
        assert len(out) == len(tokens)
        assert sum(1 for a, b in zip(tokens, out, strict=True) if a != b) == size
    elif edit == "deletion":
        assert len(out) == len(tokens) - size
        remaining = list(tokens)
        for token in out:
            remaining.remove(token)
        assert len(remaining) == size
    else:
        assert len(out) == len(tokens) + size
        added = [t for t in out if t not in set(tokens)]
        assert len(added) >= size
        assert set(added) <= set(vocab)


def test_inserted_and_substituted_words_never_come_from_the_original() -> None:
    tokens = _words(30, seed=1)
    vocab = sorted(set(tokens) | set(_words(200, seed=2)))
    out = calibrate.apply_edit(tokens, "substitution", 5, random.Random(3), vocab)
    new = [t for t in out if t not in set(tokens)]
    assert len(new) == 5
    assert not set(new) & set(tokens)


@pytest.mark.parametrize(
    ("n", "bin_"),
    [(33, "short"), (34, "medium"), (40, "medium"), (41, "long"), (200, "long")],
)
def test_length_bins_are_the_registered_cut_points(n: int, bin_: str) -> None:
    assert calibrate.length_bin(n) == bin_


def _texts(n: int) -> list[str]:
    return [" ".join(_words(14 + (i % 40), seed=1000 + i)) for i in range(n)]


def test_plan_and_build_counts_roles_and_edit_buckets() -> None:
    built = calibrate.plan_and_build(_texts(300), 60)
    roles = built["roles"]
    assert roles == {"exact": 6, "near": 36, "clean": 18, "per_bucket": 3}
    assert len(built["originals"]) == 60
    assert len(built["copies"]) == 42
    buckets = {(p["edit"], p["size"]) for p in built["plan"] if p["role"] == "near"}
    assert len(buckets) == 12
    originals = {r.id: r.fields["text"] for r in built["originals"]}
    for copy in built["copies"]:
        base_id = copy.id.replace("copy-", "orig-")
        if any(p["role"] == "exact" and p["copy_id"] == copy.id for p in built["plan"]):
            assert copy.fields["text"] == originals[base_id]
        else:
            assert copy.fields["text"] != originals[base_id]


def test_recall_stage_finds_exact_copies_on_a_small_base() -> None:
    from benchlint.schema import NearConfig

    built = calibrate.plan_and_build(_texts(300), 60)
    stored = calibrate.recall_stage(built, NearConfig())
    assert stored["exact"]["n"] == 6
    assert stored["exact"]["found"] == 6
    near_all = next(
        r
        for r in stored["rows"]
        if r["edit"] == "all" and r["size"] == "all" and r["length"] == "all"
    )
    assert near_all["n"] == 36
    assert len(stored["planted_pairs"]) == 42


def test_fpr_stage_counts_flagged_pairs_on_clean_controls() -> None:
    from benchlint.schema import NearConfig

    built = calibrate.plan_and_build(_texts(300), 60)
    fpr = calibrate.fpr_stage(built, NearConfig(), 40)
    assert fpr["pairs"] == 40
    assert fpr["flagged"] == 0
    assert fpr["exact_jaccard_at_or_above_threshold"] == 0


# --- specs ----------------------------------------------------------------------------------


def test_every_committed_spec_loads_as_an_audit_spec() -> None:
    names = sorted(p.name for p in (REPO_ROOT / "specs").glob("*.yaml"))
    assert len(names) == 7
    for name in names:
        audit = load_audit_spec(REPO_ROOT / "specs" / name)
        assert audit.dataset.source.kind == "hf"
        assert audit.dataset.source.split_map


def test_snli_and_mnli_drop_the_no_label_value_and_cap_shortcut_rows() -> None:
    for name in ("snli", "mnli"):
        audit = load_audit_spec(REPO_ROOT / "specs" / f"{name}.yaml")
        assert audit.dataset.drop_label_values == [-1]
        assert audit.dataset.max_rows_per_split == 100000
        assert audit.checks.shortcut.max_rows == 100000


def test_mbpp_spec_checks_humaneval_prompts_as_the_reference() -> None:
    audit = load_audit_spec(REPO_ROOT / "specs" / "mbpp-vs-humaneval.yaml")
    assert audit.checks.ngram.reference_fields == ["prompt"]
    assert audit.checks.ngram.reference[0].id_or_path == "openai/openai_humaneval"


def test_e3_lists_six_datasets_and_its_summary_shape() -> None:
    assert len(run_e3.SPECS) == 6
    assert set(run_e3.SMOKE_SPECS) <= set(run_e3.SPECS)


# --- README splice --------------------------------------------------------------------------


def test_splice_replaces_each_marker_pair_in_order() -> None:
    text = "a\n<!-- results:start -->\nold1\n<!-- results:end -->\nmid\n" + (
        "<!-- results:start -->\nold2\n<!-- results:end -->\nz"
    )
    out = render_readme.splice(text, ["NEW1", "NEW2"])
    assert out.index("NEW1") < out.index("mid") < out.index("NEW2")
    assert "old1" not in out and "old2" not in out
    assert out.count("<!-- results:start -->") == 2 and out.count("<!-- results:end -->") == 2


def test_splice_rejects_a_mismatched_block_count() -> None:
    text = "<!-- results:start -->\nx\n<!-- results:end -->"
    with pytest.raises(ValueError):
        render_readme.splice(text, ["a", "b"])
    with pytest.raises(ValueError):
        render_readme.splice(text + "\n<!-- results:start -->\n", ["a"])


def test_check_mode_fails_when_the_readme_is_stale(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- results:start -->\nstale\n<!-- results:end -->\n"
        "<!-- results:start -->\nstale\n<!-- results:end -->\n",
        encoding="utf-8",
    )
    empty = tmp_path / "results"
    empty.mkdir()
    assert render_readme.main(["--check", "--readme", str(readme), "--results", str(empty)]) == 1
    assert render_readme.main(["--readme", str(readme), "--results", str(empty)]) == 0
    assert render_readme.main(["--check", "--readme", str(readme), "--results", str(empty)]) == 0
    assert readme.read_text(encoding="utf-8").count(NO_RESULTS) == 2


def test_rendered_blocks_follow_the_results_present() -> None:
    """E1 says there are no results only when its calibration file is absent."""
    blocks = render_readme.render_blocks(RESULTS_DIR)
    assert len(blocks) == 2
    if not (RESULTS_DIR / "e1" / "calibration.json").is_file():
        assert NO_RESULTS in blocks[0]
    else:
        assert NO_RESULTS not in blocks[0]


# --- site -----------------------------------------------------------------------------------


def test_site_builds_with_no_results_and_says_so(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    out = tmp_path / "dist"
    written = build_site.build(empty, out)
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "No E3 audit reports yet." in index
    assert NO_RESULTS.strip("_")[:20] in index.replace("&#x27;", "'")
    assert (out / ".nojekyll") in written


def test_site_copies_only_full_mode_json(tmp_path: Path) -> None:
    import json

    results = tmp_path / "results"
    (results / "e1").mkdir(parents=True)
    (results / "e1" / "calibration.json").write_text(json.dumps({"mode": "full"}), "utf-8")
    (results / "e1" / "recall.json").write_text(json.dumps({"mode": "smoke"}), "utf-8")
    copied = build_site.copy_full_results(results, tmp_path / "dest")
    assert [p.name for p in copied] == ["calibration.json"]
