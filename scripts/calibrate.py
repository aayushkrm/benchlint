"""E1: detector calibration with planted exact and near-duplicates (docs/prereg.md, E1).

Plants known copies into 2,000 clean AG News test items and measures how often the detector
recovers them, by edit type, edit size and text length. Then measures the false-positive rate
on 10,000 random pairs of clean controls.

Usage:
    python scripts/calibrate.py            # full run, writes results/e1/
    python scripts/calibrate.py --smoke    # tiny sizes, under 60 s, writes to a temp folder
    python scripts/calibrate.py --force    # ignore stored stage files and recompute

Outputs: results/e1/recall.json, results/e1/fpr.json (stage files, resumable),
results/e1/calibration.json (the combined result) and results/e1/env.json.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import envinfo  # noqa: E402
from benchlint.checks.exact import find_exact_duplicates  # noqa: E402
from benchlint.checks.near import find_near_duplicates, jaccard, record_shingles  # noqa: E402
from benchlint.experiments import (  # noqa: E402
    LOW_N,
    MODE_FULL,
    MODE_SMOKE,
    RESULTS_DIR,
    config_hash,
    read_if_same_config,
    write_json,
)
from benchlint.io import load_dataset  # noqa: E402
from benchlint.normalize import normalize_text  # noqa: E402
from benchlint.report import wilson_interval  # noqa: E402
from benchlint.schema import DatasetSpec, NearConfig, Record, SourceSpec  # noqa: E402

SOURCE_ID = "fancyzhx/ag_news"
SPLIT = "test"
MIN_WORDS = 12
BASE_SEED = 0
ROLE_SEED = 0
EDIT_SEED = 0
FPR_SEED = 1
EDITS: tuple[str, ...] = ("substitution", "deletion", "insertion")
SIZES: tuple[int, ...] = (1, 3, 5, 10)
BUCKETS: tuple[tuple[str, int], ...] = tuple((e, s) for e in EDITS for s in SIZES)
EXACT_SHARE = 0.10
NEAR_SHARE = 0.60
FULL = {"n_base": 2000, "fpr_pairs": 10_000}
SMOKE = {"n_base": 200, "fpr_pairs": 300}
SHORT_MAX = 34
MEDIUM_MAX = 41
WORD = re.compile(r"[a-z]+")


def length_bin(n_words: int) -> str:
    """Length bin of a text from its normalised word count (short, medium or long)."""
    if n_words < SHORT_MAX:
        return "short"
    if n_words < MEDIUM_MAX:
        return "medium"
    return "long"


def load_pool() -> list[str]:
    """Texts of the eligible AG News test items, in source order (at least MIN_WORDS words)."""
    spec = DatasetSpec(
        name="ag-news-e1",
        source=SourceSpec(
            kind="hf",
            id_or_path=SOURCE_ID,
            config=None,
            revision=None,
            split_map={SPLIT: SPLIT},
        ),
        text_fields=["text"],
        label_field=None,
        id_field=None,
        max_rows_per_split=None,
        seed=BASE_SEED,
    )
    rows = load_dataset(spec)[SPLIT]
    return [r.fields["text"] for r in rows if len(r.fields["text"].split()) >= MIN_WORDS]


def apply_edit(
    tokens: list[str],
    edit: str,
    size: int,
    rng: random.Random,
    vocab: list[str],
) -> list[str]:
    """Apply ``size`` word edits at distinct positions, returning the edited token list.

    Substituted and inserted words come from ``vocab`` and never equal a token of the original,
    so the word-level edit distance is exactly ``size``.
    """
    original = set(tokens)
    used: set[str] = set()

    def new_word() -> str:
        while True:
            word = vocab[rng.randrange(len(vocab))]
            if word not in original and word not in used:
                used.add(word)
                return word

    out = list(tokens)
    if edit == "substitution":
        for pos in sorted(rng.sample(range(len(out)), size)):
            out[pos] = new_word()
        return out
    if edit == "deletion":
        for pos in sorted(rng.sample(range(len(out)), size), reverse=True):
            del out[pos]
        return out
    if edit == "insertion":
        gaps = sorted(rng.sample(range(len(tokens) + 1), size))
        result: list[str] = []
        gi = 0
        for idx in range(len(tokens) + 1):
            while gi < len(gaps) and gaps[gi] == idx:
                result.append(new_word())
                gi += 1
            if idx < len(tokens):
                result.append(tokens[idx])
        return result
    raise ValueError(f"unknown edit {edit!r}")


def plan_and_build(texts: list[str], n_base: int) -> dict[str, Any]:
    """Choose the base items and their roles, then build the planted records.

    Returns the originals, the planted copies, the role counts and one row per planted copy.
    """
    if len(texts) < n_base:
        raise ValueError(f"only {len(texts)} eligible items for a base of {n_base}")
    base = sorted(random.Random(BASE_SEED).sample(range(len(texts)), n_base))
    base_texts = [texts[i] for i in base]

    n_exact = int(n_base * EXACT_SHARE)
    per_bucket = int(n_base * NEAR_SHARE) // len(BUCKETS)
    n_near = per_bucket * len(BUCKETS)
    order = list(range(n_base))
    random.Random(ROLE_SEED).shuffle(order)
    exact_idx = sorted(order[:n_exact])
    near_order = order[n_exact : n_exact + n_near]
    clean_idx = sorted(order[n_exact + n_near :])

    vocab = sorted({w for t in texts for w in WORD.findall(t.lower()) if WORD.fullmatch(w)})
    edit_rng = random.Random(EDIT_SEED)

    originals = [
        Record(id=f"orig-{i}", split="orig", fields={"text": t}, label=None)
        for i, t in enumerate(base_texts)
    ]
    copies: list[Record] = []
    plan: list[dict[str, Any]] = []
    for i in exact_idx:
        copies.append(
            Record(id=f"copy-{i}", split="copy", fields={"text": base_texts[i]}, label=None)
        )
        plan.append(_plan_row(i, "exact", "none", 0, base_texts[i]))
    for j, i in enumerate(sorted(near_order)):
        edit, size = BUCKETS[j % len(BUCKETS)]
        tokens = base_texts[i].split()
        edited = " ".join(apply_edit(tokens, edit, size, edit_rng, vocab))
        copies.append(Record(id=f"copy-{i}", split="copy", fields={"text": edited}, label=None))
        plan.append(_plan_row(i, "near", edit, size, base_texts[i]))
    length_base = {"short": 0, "medium": 0, "long": 0}
    for t in base_texts:
        length_base[length_bin(len(normalize_text(t).split()))] += 1
    return {
        "originals": originals,
        "copies": copies,
        "plan": plan,
        "roles": {
            "exact": n_exact,
            "near": n_near,
            "clean": len(clean_idx),
            "per_bucket": per_bucket,
        },
        "length_bins": length_base,
        "clean_idx": clean_idx,
        "base_texts": base_texts,
        "eligible": len(texts),
        "vocab_size": len(vocab),
    }


def _plan_row(i: int, role: str, edit: str, size: int, original: str) -> dict[str, Any]:
    n_words = len(normalize_text(original).split())
    return {
        "base": i,
        "role": role,
        "edit": edit,
        "size": size,
        "length": length_bin(n_words),
        "n_words": n_words,
        "orig_id": f"orig-{i}",
        "copy_id": f"copy-{i}",
    }


def recall_stage(built: dict[str, Any], near_cfg: NearConfig) -> dict[str, Any]:
    """Run the detector over originals and copies; score each planted pair."""
    records: list[Record] = [*built["originals"], *built["copies"]]
    findings = find_near_duplicates(records, ["text"], near_cfg, seed=BASE_SEED)
    cross = next(f for f in findings if f.details["scope"] == "cross_split")
    if cross.details["pairs_truncated"]:
        raise RuntimeError("near pair list was truncated; raise PAIR_LIMIT before running E1")
    listed = {frozenset((a, b)) for a, b, _ in cross.details["pairs"]}

    exact_finding = next(
        f for f in find_exact_duplicates(records, ["text"]) if f.splits == ["orig", "copy"]
    )
    shingles = {r.id: record_shingles(r, ["text"], near_cfg.shingle) for r in records}

    planted: list[dict[str, Any]] = []
    for row in built["plan"]:
        pair = frozenset((row["orig_id"], row["copy_id"]))
        jac = jaccard(shingles[row["orig_id"]], shingles[row["copy_id"]])
        planted.append({**row, "found": pair in listed, "jaccard": round(jac, 6)})

    rows: list[dict[str, Any]] = []
    for edit in ("all", *EDITS):
        for size in ("all", *SIZES):
            for length in ("all", "short", "medium", "long"):
                cell = [
                    p
                    for p in planted
                    if p["role"] == "near"
                    and (edit == "all" or p["edit"] == edit)
                    and (size == "all" or p["size"] == size)
                    and (length == "all" or p["length"] == length)
                ]
                rows.append(_cell_row(edit, size, length, cell))

    exact_planted = [p for p in planted if p["role"] == "exact"]
    exact_found = exact_finding.count
    return {
        "roles": built["roles"],
        "length_bins": built["length_bins"],
        "eligible": built["eligible"],
        "vocab_size": built["vocab_size"],
        "rows": rows,
        "exact": {
            "n": len(exact_planted),
            "found": exact_found,
            **_ci(exact_found, len(exact_planted)),
        },
        "exact_check_splits": exact_finding.splits,
        "n_cross_near_pairs": len(cross.details["pairs"]),
        "n_within_near_pairs": next(
            f for f in findings if f.details["scope"] == "within_split"
        ).details["n_near_pairs"],
        "planted_pairs": [
            [
                p["orig_id"],
                p["copy_id"],
                p["role"],
                p["edit"],
                p["size"],
                p["length"],
                p["jaccard"],
                p["found"],
            ]
            for p in planted
        ],
    }


def _ci(found: int, n: int) -> dict[str, float]:
    lo, hi = wilson_interval(found, n)
    return {"recall": found / n if n else 0.0, "ci_low": lo, "ci_high": hi}


def _cell_row(edit: Any, size: Any, length: str, cell: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cell)
    found = sum(1 for p in cell if p["found"])
    jacs = sorted(p["jaccard"] for p in cell)
    median = (jacs[n // 2] if n % 2 else (jacs[n // 2 - 1] + jacs[n // 2]) / 2) if n else 0.0
    return {
        "edit": edit,
        "size": size,
        "length": length,
        "n": n,
        "found": found,
        **_ci(found, n),
        "median_jaccard": round(median, 6),
        "low_n": n < LOW_N,
    }


def fpr_stage(built: dict[str, Any], near_cfg: NearConfig, n_pairs: int) -> dict[str, Any]:
    """False-positive rate on random pairs of clean controls (no planted copy involved)."""
    controls = list(built["clean_idx"])
    texts = built["base_texts"]
    rng = random.Random(FPR_SEED)
    seen: set[tuple[int, int]] = set()
    if len(controls) * (len(controls) - 1) // 2 < n_pairs:
        raise ValueError("not enough clean controls for the requested number of pairs")
    while len(seen) < n_pairs:
        a, b = rng.sample(controls, 2)
        seen.add((min(a, b), max(a, b)))
    shingles: dict[int, Any] = {
        i: record_shingles(_rec(i, texts[i]), ["text"], near_cfg.shingle) for i in controls
    }
    flagged = 0
    oracle = 0
    histogram = [0] * 10
    max_j = 0.0
    for a, b in sorted(seen):
        ra, rb = _rec(a, texts[a], "a"), _rec(b, texts[b], "b")
        cross = next(
            f
            for f in find_near_duplicates([ra, rb], ["text"], near_cfg, seed=BASE_SEED)
            if f.details["scope"] == "cross_split"
        )
        if cross.count > 0:
            flagged += 1
        jac = jaccard(shingles[a], shingles[b])
        max_j = max(max_j, jac)
        if jac >= near_cfg.threshold:
            oracle += 1
        histogram[min(int(jac * 10), 9)] += 1
    lo, hi = wilson_interval(flagged, n_pairs)
    return {
        "pairs": n_pairs,
        "flagged": flagged,
        "rate": flagged / n_pairs,
        "ci_low": lo,
        "ci_high": hi,
        "exact_jaccard_at_or_above_threshold": oracle,
        "max_jaccard": round(max_j, 6),
        "jaccard_histogram": {
            "bin_edges": [i / 10 for i in range(11)],
            "counts": histogram,
        },
        "seed": FPR_SEED,
        "controls": len(controls),
    }


def _rec(i: int, text: str, split: str = "orig") -> Record:
    return Record(id=f"ctl-{i}", split=split, fields={"text": text}, label=None)


def main(argv: list[str] | None = None) -> int:
    """Run E1 and write the result files. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="E1 detector calibration.")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, temp output folder")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument("--force", action="store_true", help="recompute stage files")
    args = parser.parse_args(argv)

    mode = MODE_SMOKE if args.smoke else MODE_FULL
    sizes = SMOKE if args.smoke else FULL
    out = args.out
    if out is None:
        out = (
            Path(tempfile.mkdtemp(prefix="benchlint-smoke-e1-"))
            if args.smoke
            else RESULTS_DIR / "e1"
        )
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    near_cfg = NearConfig()
    settings = {
        "threshold": near_cfg.threshold,
        "num_perm": near_cfg.num_perm,
        "shingle": near_cfg.shingle,
        "lsh_threshold": near_cfg.lsh_threshold,
        "seed": BASE_SEED,
        "role_seed": ROLE_SEED,
        "edit_seed": EDIT_SEED,
        "fpr_seed": FPR_SEED,
        "exact_share": EXACT_SHARE,
        "near_share": NEAR_SHARE,
        "min_words": MIN_WORDS,
        "n_base": sizes["n_base"],
        "fpr_pairs": sizes["fpr_pairs"],
        "length_bins": {
            "short": f"under {SHORT_MAX} words",
            "medium": f"{SHORT_MAX} to {MEDIUM_MAX - 1} words",
            "long": f"{MEDIUM_MAX} words or more",
        },
        "source": SOURCE_ID,
        "split": SPLIT,
    }
    chash = config_hash(settings)

    recall_path = out / "recall.json"
    fpr_path = out / "fpr.json"
    stored = None if args.force else read_if_same_config(recall_path, chash)
    built: dict[str, Any] | None = None
    if stored is None:
        built = plan_and_build(load_pool(), sizes["n_base"])
        stored = {"config_hash": chash, "mode": mode, **recall_stage(built, near_cfg)}
        write_json(recall_path, stored)
        print(f"recall stage done: {len(built['copies'])} planted copies", file=sys.stderr)
    else:
        print("recall stage: reusing stored result", file=sys.stderr)

    fpr = None if args.force else read_if_same_config(fpr_path, chash)
    if fpr is None:
        if built is None:
            built = plan_and_build(load_pool(), sizes["n_base"])
        fpr = {
            "config_hash": chash,
            "mode": mode,
            **fpr_stage(built, near_cfg, sizes["fpr_pairs"]),
        }
        write_json(fpr_path, fpr)
        print(f"fpr stage done: {fpr['flagged']} of {fpr['pairs']} flagged", file=sys.stderr)
    else:
        print("fpr stage: reusing stored result", file=sys.stderr)

    combined = {
        "experiment": "E1",
        "mode": mode,
        "config_hash": chash,
        "prereg": "docs/prereg.md",
        "synthetic_edits": True,
        "settings": settings,
        "source": {
            "dataset": SOURCE_ID,
            "split": SPLIT,
            "n_base": sizes["n_base"],
            "eligible": stored["eligible"],
            "vocab_size": stored["vocab_size"],
            "min_words": MIN_WORDS,
        },
        "roles": stored["roles"],
        "length_bins": stored["length_bins"],
        "rows": stored["rows"],
        "exact": stored["exact"],
        "fpr": fpr,
        "runtime_s": round(time.perf_counter() - start, 3),
    }
    write_json(out / "calibration.json", combined)
    write_json(out / "env.json", envinfo.collect())
    print(f"wrote {out / 'calibration.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
