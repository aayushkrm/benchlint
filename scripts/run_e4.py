"""E4: dogfooding. benchlint applied to two sibling projects' inputs (docs/prereg.md, E4).

Inputs:
- quantgate: the committed ARC-Easy and GSM8K eval subsets (IDs only). The text comes from the
  Hugging Face datasets through benchlint's loader. Each eval item is checked against the train
  split (exact and near duplicates) on the text quantgate's prompt contains.
- passk: the 24 task instructions (field ``instruction``). Checked within the set, and every
  pairwise 5-gram Jaccard is summarised.

Sibling checkouts are read in place. Results record each sibling's git commit, dirty flag and
changed paths (relative), and the SHA-256 of each input file. No absolute path is written.

Usage:
    python scripts/run_e4.py                                  # default ../quantgate ../passk
    python scripts/run_e4.py --quantgate PATH --passk PATH --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import yaml  # noqa: E402

import envinfo  # noqa: E402
from benchlint.checks.exact import find_exact_duplicates  # noqa: E402
from benchlint.checks.near import (  # noqa: E402
    find_near_duplicates,
    jaccard,
    record_shingles,
)
from benchlint.experiments import (  # noqa: E402
    MODE_FULL,
    REPO_ROOT,
    RESULTS_DIR,
    write_json,
)
from benchlint.io import load_dataset_detailed  # noqa: E402
from benchlint.report import wilson_interval  # noqa: E402
from benchlint.schema import DatasetSpec, ExamplePair, NearConfig, Record, SourceSpec  # noqa: E402

DEFAULT_QUANTGATE = REPO_ROOT.parent / "quantgate"
DEFAULT_PASSK = REPO_ROOT.parent / "passk"
ARC_SUBSET = "data/subsets/arc_easy_test.json"
GSM8K_SUBSET = "data/subsets/gsm8k_test.json"
PASSK_TASKS = "src/passk/tasks/labstock"
TEXT_LIMIT = 200
EXAMPLE_LIMIT = 5
PAIRWISE_THRESHOLD = 0.8
HIST_BINS = 10


class E4Error(RuntimeError):
    """A missing input or a failed check. The run stops; nothing is written silently."""


# --- provenance ------------------------------------------------------------------------------


def _git(path: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def sibling_provenance(name: str, path: Path) -> dict[str, Any]:
    """Commit, dirty flag and changed paths of a sibling checkout. Relative paths only."""
    if not path.is_dir():
        raise E4Error(f"sibling checkout not found: {name}")
    sha = (_git(path, "rev-parse", "HEAD") or "").strip() or None
    status = _git(path, "status", "--porcelain")
    if status is None:
        dirty: bool | None = None
        paths: list[str] | None = None
    else:
        paths = sorted({line[3:] for line in status.splitlines() if line.strip()})
        dirty = bool(paths)
    return {"name": name, "git_sha": sha, "git_dirty": dirty, "git_dirty_paths": paths}


def file_record(root: Path, relative: str) -> dict[str, str]:
    """A relative path and the SHA-256 of the file at it."""
    target = root / relative
    if not target.is_file():
        raise E4Error(f"input file not found: {relative}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"path": relative, "sha256": digest}


# --- quantgate: ARC-Easy and GSM8K against their train splits -------------------------------


def load_ids(root: Path, relative: str) -> dict[str, Any]:
    """The committed subset: its selection metadata and its item IDs."""
    data = json.loads((root / relative).read_text(encoding="utf-8"))
    ids = [str(x) for x in data["ids"]]
    if len(set(ids)) != len(ids):
        raise E4Error(f"duplicate IDs in {relative}")
    return {"ids": ids, "n": data["n"], "seed": data["seed"], "selection": data["selection"]}


def arc_text(fields: dict[str, str]) -> str:
    """Question plus choice texts, joined by spaces: the text quantgate's ARC prompt holds."""
    choices = json.loads(fields["choices"])["text"]
    return " ".join([fields["question"], *choices])


def arc_question_text(fields: dict[str, str]) -> str:
    """The question alone: the text the E3 ARC spec uses. Supplementary check only."""
    return fields["question"]


def gsm8k_question_and_answer(fields: dict[str, str]) -> str:
    """Question and answer: the text the E3 GSM8K spec uses. Supplementary check only."""
    return f"{fields['question']} {fields['answer']}"


def gsm8k_text(fields: dict[str, str]) -> str:
    """The question: quantgate's GSM8K prompt does not contain the gold answer."""
    return fields["question"]


def _hf_spec(
    name: str, dataset: str, config: str, id_field: str | None, label: str | None, text_field: str
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        source=SourceSpec(
            kind="hf",
            id_or_path=dataset,
            config=config,
            revision=None,
            split_map={"train": "train", "test": "test"},
        ),
        text_fields=[text_field],
        label_field=label,
        id_field=id_field,
        max_rows_per_split=None,
        seed=0,
    )


def eval_records_from(by_key: dict[str, Record], text_of: Any, ids: list[str]) -> list[Record]:
    """Committed eval items in committed order. A missing ID stops the run; none is dropped."""
    missing = [i for i in ids if i not in by_key]
    if missing:
        raise E4Error(f"{len(missing)} committed IDs not found in the test split: {missing[:5]}")
    return [
        Record(id=f"test:{i}", split="test", fields={"text": text_of(by_key[i].fields)}, label=None)
        for i in ids
    ]


def quantgate_records(
    spec: DatasetSpec, key_of: Any, text_of: Any, ids: list[str]
) -> tuple[list[Record], list[Record], int]:
    """Train records (every row) and the committed eval records, on the prompt text."""
    splits, _ = load_dataset_detailed(spec)
    train = [
        Record(
            id=f"train:{key_of(r)}",
            split="train",
            fields={"text": text_of(r.fields)},
            label=None,
        )
        for r in splits["train"]
    ]
    by_key = {key_of(r): r for r in splits["test"]}
    evals = eval_records_from(by_key, text_of, ids)
    return train, evals, len(splits["train"])


def _rate(count: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(count, n)
    return {"n": n, "matched": count, "rate": count / n if n else 0.0, "ci_low": lo, "ci_high": hi}


def _examples(items: Sequence[ExamplePair]) -> list[dict[str, Any]]:
    return [e.model_dump(mode="json") for e in items]


def eval_matches(records: list[Record], eval_ids: set[str], config: NearConfig) -> dict[str, Any]:
    """Exact and near matches of eval items against train. Only the cross-split direction counts.

    Train records come first, so the cross finding lists (train, eval) partners. Exact matches
    are a subset of near matches (identical text has identical shingles).
    """
    exact_cross = next(
        f for f in find_exact_duplicates(records, ["text"]) if f.splits == ["train", "test"]
    )
    near_cross = next(
        f
        for f in find_near_duplicates(records, ["text"], config, seed=0)
        if f.details["scope"] == "cross_split"
    )
    if exact_cross.details.get("pairs_truncated") or near_cross.details["pairs_truncated"]:
        raise E4Error("pair list truncated; E4 expects every pair to be listed")
    exact_hit = {b for _, b, _ in exact_cross.details["pairs"]} & eval_ids
    near_hit = {
        x for a, b, _ in near_cross.details["pairs"] for x in (a, b) if x.startswith("test:")
    } & eval_ids
    n = len(eval_ids)
    return {
        "exact": {**_rate(len(exact_hit), n), "examples": _examples(exact_cross.examples)},
        "near": {
            **_rate(len(near_hit), n),
            "threshold": config.threshold,
            "examples": _examples(near_cross.examples),
            "candidate_jaccard_histogram": near_cross.details["jaccard_histogram"],
        },
    }


def run_quantgate(
    root: Path,
    sub: str,
    dataset: str,
    config_name: str,
    key_of: Any,
    text_of: Any,
    text_field: str,
    label: str | None,
    id_field: str | None,
    name: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    subset = load_ids(root, sub)
    spec = _hf_spec(name, dataset, config_name, id_field, label, text_field)
    train, evals, n_train_source = quantgate_records(spec, key_of, text_of, subset["ids"])
    records = [*train, *evals]
    eval_ids = {r.id for r in evals}
    matches = eval_matches(records, eval_ids, NearConfig())
    return {
        "experiment": "E4",
        "mode": MODE_FULL,
        "dataset": name,
        "sibling": "quantgate",
        "source": dataset,
        "config": config_name,
        "eval_split": "test",
        "train_split": "train",
        "n_train": len(train),
        "n_train_source": n_train_source,
        "n_eval": len(evals),
        "subset": {
            "file": sub,
            "n_committed": subset["n"],
            "selection": subset["selection"],
            "seed": subset["seed"],
        },
        "text": "quantgate prompt text (ARC: question and choice texts; GSM8K: question)",
        "settings": {
            "threshold": NearConfig().threshold,
            "num_perm": NearConfig().num_perm,
            "shingle": NearConfig().shingle,
            "lsh_threshold": NearConfig().lsh_threshold,
            "seed": 0,
        },
        "checks": matches,
        "runtime_s": round(time.perf_counter() - start, 3),
    }


def arc_key(record: Record) -> str:
    return record.id


def gsm8k_key(record: Record) -> str:
    return record.id.split(":", 1)[1]


def _supplementary(main: dict[str, Any], text: str, checks: dict[str, Any]) -> dict[str, Any]:
    """A check on the same subset with another text. Not pre-registered: reported as such."""
    return {
        "pre_registered": False,
        "text": text,
        "n_eval": main["n_eval"],
        "n_train": main["n_train"],
        "checks": checks,
    }


def run_arc(root: Path) -> dict[str, Any]:
    main = run_quantgate(
        root,
        ARC_SUBSET,
        "allenai/ai2_arc",
        "ARC-Easy",
        arc_key,
        arc_text,
        "question",
        "answerKey",
        "id",
        "arc-easy",
    )
    supp = run_quantgate(
        root,
        ARC_SUBSET,
        "allenai/ai2_arc",
        "ARC-Easy",
        arc_key,
        arc_question_text,
        "question",
        "answerKey",
        "id",
        "arc-easy",
    )
    main["supplementary"] = _supplementary(
        main, "supplementary, not pre-registered: question only (the E3 ARC text)", supp["checks"]
    )
    return main


def run_gsm8k(root: Path) -> dict[str, Any]:
    main = run_quantgate(
        root,
        GSM8K_SUBSET,
        "openai/gsm8k",
        "main",
        gsm8k_key,
        gsm8k_text,
        "question",
        None,
        None,
        "gsm8k",
    )
    supp = run_quantgate(
        root,
        GSM8K_SUBSET,
        "openai/gsm8k",
        "main",
        gsm8k_key,
        gsm8k_question_and_answer,
        "question",
        None,
        None,
        "gsm8k",
    )
    main["supplementary"] = _supplementary(
        main,
        "supplementary, not pre-registered: question and answer (the E3 GSM8K text)",
        supp["checks"],
    )
    return main


# --- passk: the 24 instructions ------------------------------------------------------------


def load_instructions(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(id, instruction) pairs for every task file, and the file records for provenance."""
    folder = root / PASSK_TASKS
    if not folder.is_dir():
        raise E4Error("passk task folder not found")
    files = sorted(folder.glob("*.yaml"))
    tasks: list[dict[str, str]] = []
    records: list[dict[str, str]] = []
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("instruction"):
            raise E4Error(f"task file without an instruction: {path.name}")
        tasks.append(
            {"id": str(data.get("id", path.stem)), "instruction": str(data["instruction"])}
        )
        records.append(file_record(root, f"{PASSK_TASKS}/{path.name}"))
    return tasks, records


def pairwise_summary(records: list[Record]) -> dict[str, Any]:
    """Exact 5-gram Jaccard for every pair, summarised; the top pairs with excerpts."""
    shingles = {r.id: record_shingles(r, ["text"], 5) for r in records}
    ids = [r.id for r in records]
    values: list[float] = []
    pairs: list[tuple[float, str, str]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            j = jaccard(shingles[a], shingles[b])
            values.append(j)
            pairs.append((j, a, b))
    ordered = sorted(values)
    m = len(ordered)
    median = (
        (ordered[m // 2] if m % 2 else (ordered[m // 2 - 1] + ordered[m // 2]) / 2) if m else 0.0
    )
    counts = [0] * HIST_BINS
    for v in values:
        counts[min(int(v * HIST_BINS), HIST_BINS - 1)] += 1
    above = sum(1 for v in values if v >= PAIRWISE_THRESHOLD)
    text_of = {r.id: r.fields["text"] for r in records}
    top = sorted(pairs, key=lambda t: (-t[0], t[1], t[2]))[:EXAMPLE_LIMIT]
    return {
        "pairs": m,
        "min": ordered[0] if m else 0.0,
        "median": median,
        "max": ordered[-1] if m else 0.0,
        "histogram": {"bin_edges": [i / HIST_BINS for i in range(HIST_BINS + 1)], "counts": counts},
        "at_or_above_threshold": _rate(above, m) | {"threshold": PAIRWISE_THRESHOLD},
        "top_pairs": [
            {
                "a_id": a,
                "b_id": b,
                "jaccard": round(j, 6),
                "a_text": text_of[a][:TEXT_LIMIT],
                "b_text": text_of[b][:TEXT_LIMIT],
            }
            for j, a, b in top
        ],
    }


def run_passk(root: Path) -> dict[str, Any]:
    start = time.perf_counter()
    tasks, file_records = load_instructions(root)
    records = [
        Record(id=f"tasks:{t['id']}", split="tasks", fields={"text": t["instruction"]}, label=None)
        for t in tasks
    ]
    config = NearConfig()
    exact = next(f for f in find_exact_duplicates(records, ["text"]) if f.splits == ["tasks"])
    near = next(
        f
        for f in find_near_duplicates(records, ["text"], config, seed=0)
        if f.details["scope"] == "within_split"
    )
    if near.details["pairs_truncated"]:
        raise E4Error("near pair list truncated")
    near_pairs = {frozenset((a, b)) for a, b, _ in near.details["pairs"]}
    pairwise = pairwise_summary(records)
    shingles = {r.id: record_shingles(r, ["text"], 5) for r in records}
    matrix_above = {
        frozenset((a.id, b.id))
        for i, a in enumerate(records)
        for b in records[i + 1 :]
        if jaccard(shingles[a.id], shingles[b.id]) >= PAIRWISE_THRESHOLD
    }
    n = len(records)
    instructions_hash = hashlib.sha256(
        "\n".join(t["instruction"] for t in tasks).encode("utf-8")
    ).hexdigest()
    return {
        "experiment": "E4",
        "mode": MODE_FULL,
        "dataset": "passk-instructions",
        "sibling": "passk",
        "field": "instruction",
        "n_tasks": n,
        "word_range": [
            min(len(t["instruction"].split()) for t in tasks),
            max(len(t["instruction"].split()) for t in tasks),
        ],
        "files": file_records,
        "instructions_sha256": instructions_hash,
        "settings": {
            "threshold": config.threshold,
            "num_perm": config.num_perm,
            "shingle": config.shingle,
            "lsh_threshold": config.lsh_threshold,
            "seed": 0,
        },
        "exact": {
            **_rate(sum(len(g) - 1 for g in _groups(exact)), n),
            "groups": len(_groups(exact)),
            "examples": _examples(exact.examples),
        },
        "near": {
            **_rate(len({i for pair in near_pairs for i in pair}), n),
            "threshold": config.threshold,
            "examples": _examples(near.examples),
        },
        "lsh_missed_pairs_at_threshold": len(matrix_above - near_pairs),
        "pairwise": pairwise,
        "runtime_s": round(time.perf_counter() - start, 3),
    }


def _groups(exact: Any) -> list[list[str]]:
    """Duplicate groups from the exact finding's listed pairs (connected by identical text)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b, _ in exact.details["pairs"]:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return [sorted(g) for g in groups.values() if len(g) > 1]


# --- driver ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E4 dogfooding on sibling inputs.")
    parser.add_argument("--quantgate", type=Path, default=DEFAULT_QUANTGATE)
    parser.add_argument("--passk", type=Path, default=DEFAULT_PASSK)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "e4")
    args = parser.parse_args(argv)
    try:
        qg = args.quantgate.resolve()
        pk = args.passk.resolve()
        provenance = {
            "quantgate": sibling_provenance("quantgate", qg),
            "passk": sibling_provenance("passk", pk),
        }
        provenance["quantgate"]["subset_files"] = [
            file_record(qg, ARC_SUBSET),
            file_record(qg, GSM8K_SUBSET),
        ]
        arc = run_arc(qg)
        gsm8k = run_gsm8k(qg)
        passk = run_passk(pk)
    except E4Error as exc:
        print(f"E4 failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "experiment": "E4",
        "mode": MODE_FULL,
        "scope": "dogfooding: sibling inputs, small n",
        "siblings": provenance,
        "datasets": {"quantgate_arc_easy": arc, "quantgate_gsm8k": gsm8k, "passk": passk},
        "prereg": "docs/prereg.md",
    }
    write_json(args.out / "quantgate_arc_easy.json", arc)
    write_json(args.out / "quantgate_gsm8k.json", gsm8k)
    write_json(args.out / "passk_instructions.json", passk)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "env.json", envinfo.collect())
    print(
        f"E4: ARC near {arc['checks']['near']['matched']}/{arc['n_eval']}, "
        f"GSM8K near {gsm8k['checks']['near']['matched']}/{gsm8k['n_eval']}, "
        f"passk near {passk['near']['matched']}/{passk['n_tasks']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
