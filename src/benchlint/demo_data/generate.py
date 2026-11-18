"""Deterministic generator for the SYNTHETIC demo dataset. No downloads, no real text.

Layout written under ``out_dir``:

- ``data/train.jsonl`` and ``data/test.jsonl``: one row per item with ``id``, ``headline``,
  ``body`` and ``label``. The dataset spec points at the directory ``data`` and maps the splits
  ``train`` and ``test`` to these file names.
- ``reference/reference.jsonl``: 150 reference documents (``text``) for the n-gram check.
- ``audit.yaml``: an AuditSpec that audits the files above (paths relative to ``out_dir``).
- ``manifest.json``: every planted category, with its pairs and file checksums.

Planted leaks (counts are constants below):
- exact duplicates across splits (test items copied from train),
- exact duplicates within train, of which some carry a flipped label (label conflicts),
- near-duplicates of train items in test, with contiguous 1, 3 and 5-word substitutions,
  deletions and insertions in bodies of 110 to 130 words,
- test items quoted verbatim (40 words) in reference documents (n-gram contamination),
- a headline token that agrees with the label 90% of the time (shortcut).

Everything produced here is synthetic and labelled SYNTHETIC in any output.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from benchlint.demo_data.manifest import SYNTHETIC_NAME, DemoFile, DemoManifest, PlantedLeak
from benchlint.schema import (
    AuditSpec,
    CheckConfig,
    DatasetSpec,
    NgramConfig,
    ShortcutConfig,
    SourceSpec,
    View,
)

LABELS: tuple[str, ...] = ("A", "B", "C", "D")
LABEL_TOKENS: dict[str, str] = {"A": "qzxalpha", "B": "qzxbeta", "C": "qzxgamma", "D": "qzxdelta"}
N_TRAIN = 1000
N_TEST = 300
N_REFERENCE = 150
VOCAB_SIZE = 3000
HEADLINE_WORDS = 6
BODY_MIN = 110
BODY_MAX = 130
SHORTCUT_RATE = 0.9
N_CROSS_EXACT = 12
N_WITHIN_EXACT = 8
N_CONFLICT = 6
NEAR_SIZES: tuple[int, ...] = (1, 3, 5)
NEAR_KINDS: tuple[str, ...] = ("sub", "del", "ins")
NEAR_PER_BUCKET = 3
N_NGRAM = 10
NGRAM_SPAN = 40
NGRAM_OFFSET = 10
REFERENCE_TEXT_FIELD = "text"

# Index layout inside the generated splits (kept disjoint so that each planted category is
# clean). Train sources: 100..111 for exact cross copies, 0..7 for within-train copies (written
# to 900..907), 200..226 for near-duplicate sources. Test: 0..11 exact copies, 12..38 near
# duplicates, 39..48 quoted into the reference corpus.
CROSS_SRC_START = 100
WITHIN_SRC_START = 0
WITHIN_COPY_START = 900
NEAR_SRC_START = 200
NGRAM_TEST_START = 39
TEST_NEAR_START = N_CROSS_EXACT
NEAR_TOTAL = len(NEAR_SIZES) * len(NEAR_KINDS) * NEAR_PER_BUCKET


@dataclass(frozen=True)
class _Row:
    """One synthetic item. ``headline`` and ``body`` are word tuples; rendering happens later."""

    id: str
    split: str
    headline: tuple[str, ...]
    body: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class DemoBundle:
    """What ``build_demo`` wrote: the audit spec to run, the manifest, and the output folder."""

    spec: AuditSpec
    manifest: DemoManifest
    out_dir: Path


def _make_vocab(rng: random.Random, size: int) -> list[str]:
    """Pseudo-words from consonant-vowel syllables. Sorted, so the result is independent of set
    iteration order. Never contains ``q``, so it cannot collide with the planted tokens.
    """
    consonants = "bcdfghjklmnprstvw"
    vowels = "aeiou"
    words: set[str] = set()
    while len(words) < size:
        n_syllables = rng.choice((2, 3))
        words.add("".join(rng.choice(consonants) + rng.choice(vowels) for _ in range(n_syllables)))
    return sorted(words)


def _random_row(rng: random.Random, vocab: list[str], row_id: str, split: str) -> _Row:
    """A fresh item. The shortcut token matches its label with probability ``SHORTCUT_RATE``."""
    label = rng.choice(LABELS)
    if rng.random() < SHORTCUT_RATE:
        token_label = label
    else:
        token_label = rng.choice([x for x in LABELS if x != label])
    headline = [rng.choice(vocab) for _ in range(HEADLINE_WORDS)]
    headline.insert(rng.randrange(HEADLINE_WORDS + 1), LABEL_TOKENS[token_label])
    body_len = rng.randint(BODY_MIN, BODY_MAX)
    body = [rng.choice(vocab) for _ in range(body_len)]
    return _Row(row_id, split, tuple(headline), tuple(body), label)


def _other_label(rng: random.Random, label: str) -> str:
    """A label different from ``label``, drawn from ``LABELS``."""
    return rng.choice([x for x in LABELS if x != label])


def _edit_body(
    rng: random.Random, body: tuple[str, ...], kind: str, size: int, vocab: list[str]
) -> tuple[str, ...]:
    """Apply one contiguous edit of ``size`` words: substitution, deletion or insertion."""
    words = list(body)
    start = rng.randrange(NGRAM_OFFSET, len(words) - NGRAM_OFFSET - size)
    fresh = [rng.choice(vocab) for _ in range(size)]
    if kind == "sub":
        words[start : start + size] = fresh
    elif kind == "del":
        del words[start : start + size]
    elif kind == "ins":
        words[start:start] = fresh
    else:
        raise ValueError(f"unknown edit kind: {kind}")
    return tuple(words)


def _render_body(words: tuple[str, ...]) -> str:
    """Join words and add a full stop after every 15th word (stripped by normalisation)."""
    parts = [f"{w}." if (i + 1) % 15 == 0 else w for i, w in enumerate(words)]
    return " ".join(parts)


def _token_label(row: _Row) -> str:
    """The label that the headline token of ``row`` points to."""
    for label, token in LABEL_TOKENS.items():
        if token in row.headline:
            return label
    raise ValueError(f"row {row.id} has no shortcut token")


def _row_json(row: _Row) -> dict[str, str]:
    return {
        "id": row.id,
        "headline": " ".join(row.headline),
        "body": _render_body(row.body),
        "label": row.label,
    }


def _write_jsonl(out_dir: Path, rel: str, rows: list[dict[str, str]]) -> DemoFile:
    """Write rows as sorted-key JSON lines and return the file's row count and checksum."""
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return DemoFile(
        path=Path(rel).as_posix(),
        rows=len(rows),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _audit_spec(seed: int) -> AuditSpec:
    """The AuditSpec for the demo. Paths are relative to the output folder."""
    dataset = DatasetSpec(
        name=SYNTHETIC_NAME,
        source=SourceSpec(
            kind="jsonl",
            id_or_path="data",
            config=None,
            revision=None,
            split_map={"train": "train", "test": "test"},
        ),
        text_fields=["headline", "body"],
        label_field="label",
        id_field="id",
        max_rows_per_split=None,
        seed=seed,
    )
    reference = SourceSpec(
        kind="jsonl",
        id_or_path="reference",
        config=None,
        revision=None,
        split_map={"reference": "reference"},
    )
    checks = CheckConfig(
        exact=True,
        labels=True,
        ngram=NgramConfig(n=13, reference=[reference]),
        shortcut=ShortcutConfig(
            views=[
                View(name="headline_only", kind="text_fields", fields=["headline"]),
                View(name="body_length", kind="length", fields=["body"]),
            ]
        ),
        canary=False,
    )
    return AuditSpec(dataset=dataset, checks=checks)


def build_demo(out_dir: Path, *, seed: int = 0) -> DemoBundle:
    """Write the SYNTHETIC dataset, reference corpus, audit spec and manifest to ``out_dir``.

    The output depends only on ``seed``: the same seed gives byte-identical files. Returns the
    AuditSpec (relative to ``out_dir``), the manifest of planted leaks and the output folder.
    """
    rng = random.Random(seed)
    vocab = _make_vocab(rng, VOCAB_SIZE)
    train = [_random_row(rng, vocab, f"train-{i:05d}", "train") for i in range(N_TRAIN)]
    test = [_random_row(rng, vocab, f"test-{i:05d}", "test") for i in range(N_TEST)]
    leaks: list[PlantedLeak] = []

    # Exact duplicates across splits: test[j] is a copy of train[CROSS_SRC_START + j].
    cross_pairs: list[tuple[str, str]] = []
    for j in range(N_CROSS_EXACT):
        src = train[CROSS_SRC_START + j]
        copy = replace(src, id=f"test-{j:05d}", split="test")
        test[j] = copy
        cross_pairs.append((src.id, copy.id))
    leaks.append(
        PlantedLeak(
            category="exact_cross_split",
            check="exact",
            description="test items that are exact copies of train items",
            planted=len(cross_pairs),
            splits=["train", "test"],
            pairs=cross_pairs,
        )
    )

    # Exact duplicates within train. Copies with index below N_CONFLICT get a different label.
    within_pairs: list[tuple[str, str]] = []
    conflict_pairs: list[tuple[str, str]] = []
    for j in range(N_WITHIN_EXACT):
        src = train[WITHIN_SRC_START + j]
        label = _other_label(rng, src.label) if j < N_CONFLICT else src.label
        copy = replace(src, id=f"train-{WITHIN_COPY_START + j:05d}", label=label)
        train[WITHIN_COPY_START + j] = copy
        within_pairs.append((src.id, copy.id))
        if label != src.label:
            conflict_pairs.append((src.id, copy.id))
    leaks.append(
        PlantedLeak(
            category="exact_within_train",
            check="exact",
            description="train items duplicated within train",
            planted=len(within_pairs),
            splits=["train"],
            pairs=within_pairs,
        )
    )
    leaks.append(
        PlantedLeak(
            category="label_conflict",
            check="labels",
            description="within-train duplicate pairs whose two labels differ",
            planted=len(conflict_pairs),
            splits=["train"],
            pairs=conflict_pairs,
        )
    )

    # Near-duplicates: test[TEST_NEAR_START + n] is an edited copy of train[NEAR_SRC_START + n].
    near_buckets: list[tuple[str, int]] = [
        (kind, size) for size in NEAR_SIZES for kind in NEAR_KINDS
    ]
    near_pairs: dict[int, list[tuple[str, str]]] = {size: [] for size in NEAR_SIZES}
    for n in range(NEAR_TOTAL):
        kind, size = near_buckets[n // NEAR_PER_BUCKET]
        src = train[NEAR_SRC_START + n]
        edited = replace(
            src,
            id=f"test-{TEST_NEAR_START + n:05d}",
            split="test",
            body=_edit_body(rng, src.body, kind, size, vocab),
        )
        test[TEST_NEAR_START + n] = edited
        near_pairs[size].append((src.id, edited.id))
    for size in NEAR_SIZES:
        leaks.append(
            PlantedLeak(
                category=f"near_edit_{size}word",
                check="near",
                description=(
                    f"test items edited from train by a contiguous {size}-word substitution, "
                    "deletion or insertion in the body"
                ),
                planted=len(near_pairs[size]),
                splits=["train", "test"],
                pairs=near_pairs[size],
                details={"kinds": list(NEAR_KINDS)},
            )
        )

    # Reference corpus: N_NGRAM documents quote a 40-word span of a test item.
    reference_rows: list[dict[str, str]] = []
    ngram_pairs: list[tuple[str, str]] = []
    for i in range(N_REFERENCE):
        words = [rng.choice(vocab) for _ in range(rng.randint(80, 160))]
        if i < N_NGRAM:
            src = test[NGRAM_TEST_START + i]
            span = list(src.body[NGRAM_OFFSET : NGRAM_OFFSET + NGRAM_SPAN])
            pos = rng.randrange(len(words) + 1)
            words = words[:pos] + span + words[pos:]
            ngram_pairs.append((src.id, f"ref-{i:05d}"))
        reference_rows.append({"id": f"ref-{i:05d}", REFERENCE_TEXT_FIELD: " ".join(words)})
    leaks.append(
        PlantedLeak(
            category="ngram_reference",
            check="ngram",
            description="test items quoted (40 words) in reference documents",
            planted=len(ngram_pairs),
            splits=["test", "reference"],
            pairs=ngram_pairs,
            details={"n": 13, "span_words": NGRAM_SPAN},
        )
    )

    # Shortcut: fraction of test headlines whose token matches the label.
    agree_test = sum(1 for row in test if _token_label(row) == row.label)
    leaks.append(
        PlantedLeak(
            category="shortcut_token",
            check="shortcut",
            description=(
                "a label-specific token in the headline, matching the label 90% of the time"
            ),
            planted=1,
            splits=["train", "test"],
            details={
                "view": "headline_only",
                "nominal_rate": SHORTCUT_RATE,
                "observed_agree_test": agree_test,
                "observed_rate_test": agree_test / N_TEST,
                "tokens": dict(LABEL_TOKENS),
            },
        )
    )

    files = [
        _write_jsonl(out_dir, "data/train.jsonl", [_row_json(r) for r in train]),
        _write_jsonl(out_dir, "data/test.jsonl", [_row_json(r) for r in test]),
        _write_jsonl(out_dir, "reference/reference.jsonl", reference_rows),
    ]
    spec = _audit_spec(seed)
    (out_dir / "audit.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    manifest = DemoManifest(seed=seed, files=files, leaks=leaks)
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return DemoBundle(spec=spec, manifest=manifest, out_dir=out_dir)
