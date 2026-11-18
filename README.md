# benchlint

A linter for ML benchmarks. It finds train/test leakage, exact and near-duplicates, label conflicts, n-gram contamination and shortcut features before you trust a benchmark score.

> **Status: v0.1.0 candidate. No full results are committed yet.** The checks, the CLI, the pre-registered experiments E1 to E3 and their smoke tests exist. The full experiment queue has not been run, so the results sections below say "No results yet". Nothing is published (no GitHub repository, no release, no Pages site) until the author confirms.

## Why

Benchmark scores are only as good as the split they are measured on. A test item that also appears in training data, or a near-copy of one, inflates the number, and a single feature can predict the label without any reasoning. benchlint runs these checks on a dataset spec before the score is quoted. It is a linter, so it reports what it found and how sure it is; it does not decide whether a benchmark is good.

## Quickstart

Install from a local checkout (the repository is not published yet):

```bash
git clone <this repository> benchlint && cd benchlint
uv tool install .            # or: uv sync --extra dev, then uv run benchlint ...
benchlint demo               # SYNTHETIC data with planted leaks; under 30 s; no downloads
```

`benchlint demo` exits with code 1 by default because the planted leaks are findings. Use `benchlint demo --fail-on none` to exit 0. `make demo` uses `--fail-on none`.

Audit a real dataset (needs network; the Hugging Face Hub is read through `datasets`):

```bash
benchlint audit specs/gsm8k.yaml --out reports/ --fail-on high
```

## Command line

| Command | What it does |
|---|---|
| `benchlint demo` | Audits the bundled SYNTHETIC dataset with planted leaks and prints planted vs found. |
| `benchlint audit SPEC... --out DIR` | Runs every enabled check and writes `report.json`, `report.md` and `report.html`. |
| `benchlint check exact SPEC` | Exact duplicates within and across splits. |
| `benchlint check near SPEC [--threshold T]` | Near-duplicates (MinHash LSH, verified by exact Jaccard). |
| `benchlint check ngram SPEC [--reference REF] [--n N]` | 13-gram overlap between eval items and a reference corpus. |
| `benchlint check shortcut SPEC` | Partial-input baselines (single field, length, numeric scan) against majority and chance. |
| `benchlint canary add IN OUT` / `canary scan GUID PATHS...` | Insert a GUID canary into a file; grep corpora for it. |

Every command takes `--json`. With `--json`, stdout carries one JSON object: the report or findings on success, or `{"error": ..., "exit_code": 2}` on a usage or runtime error.

Exit codes: 0 pass; 1 a finding with matches (or a flagged shortcut) is at or above `--fail-on` (`info`, `low`, `medium`, `high`, or `none` to never fail on findings); 2 usage or runtime error.

Specs are YAML. An AuditSpec has a `dataset:` mapping (source, text and label fields, caps, seed, optional `drop_label_values`) and an optional `checks:` mapping. A bare DatasetSpec is also accepted and gets the default checks. The seven committed specs are in `specs/`.

## How it works

- **Normalisation.** Exact keys use NFKC, casefold and punctuation removal per field, then a blake2b digest. Near-duplicate, label and n-gram checks tokenise words with punctuation turned into spaces, so "well-known" is two words. The choice is recorded in `docs/design.md`.
- **Exact duplicates.** Within each split and across each split pair, by key.
- **Near-duplicates.** Word 5-gram shingles, 128 MinHash permutations and LSH retrieval at 0.7. Every candidate is checked with exact Jaccard, and a pair is reported only at 0.8 or above. Each finding lists its pairs with their Jaccard.
- **Label conflicts.** Duplicate or near-duplicate groups whose members carry different labels.
- **Decontamination.** 13-gram overlap of eval items with reference texts.
- **Shortcuts.** Partial-input baselines are compared with the majority class and with chance. A view is flagged when the lower bound of its bootstrap interval exceeds the majority baseline by a margin. A flag is a prompt to investigate.
- **Reports.** Every rate carries its denominator and a 95% Wilson interval, in the Markdown, HTML and JSON reports.

## Calibration: how well the detectors work (E1)

This comes before any finding, on purpose. It says how often the detectors recover planted copies and how often they fire on clean pairs. A finding is only as good as this table.

**Read the overall recall with its cause.** Five-word shingles at a threshold of 0.8 need the copy to keep most of its shingles. One word edit changes at most five shingles, so a text of n words keeps at least (n - 4 - 5) / (n - 4 + 5) of its Jaccard. A single edit reaches 0.8 only for texts of about 49 words or more, and a few edits need far longer texts. Items in the short and medium length bins (under 41 words) are below that length, so many of their planted near-duplicates have a true Jaccard below 0.8; the true-Jaccard table shows how many. The detector cannot report those pairs by design. The table labelled "conditional on true Jaccard" separates the two: it shows how often the retrieval step finds the pairs that are allowed to be reported. The bound table gives the arithmetic. Lower the shingle size or the threshold to catch shorter edits, and measure the effect with E1 before quoting it.

<!-- results:start -->
Source: `results/e1/calibration.json` (mode full, git `6172c97baf`, environment in `results/e1/env.json`). Pre-registered in `docs/prereg.md`, E1.

**SYNTHETIC: the planted copies are generated edits of real AG News text. They measure the detector on known edits, not on real leaks.**

### E1 recall by edit type and size

| Edit | Size (words) | Found / planted | Recall [95% CI] | Median exact Jaccard |
|---|---|---|---|---|
| substitution | 1 | 22/100 | 22.0% [15.0%, 31.1%] | 0.756 |
| substitution | 3 | 0/100 | 0.0% [0.0%, 3.7%] | 0.489 |
| substitution | 5 | 0/100 | 0.0% [0.0%, 3.7%] | 0.321 |
| substitution | 10 | 0/100 | 0.0% [0.0%, 3.7%] | 0.114 |
| deletion | 1 | 35/100 | 35.0% [26.4%, 44.7%] | 0.780 |
| deletion | 3 | 1/100 | 1.0% [0.2%, 5.4%] | 0.500 |
| deletion | 5 | 0/100 | 0.0% [0.0%, 3.7%] | 0.368 |
| deletion | 10 | 0/100 | 0.0% [0.0%, 3.7%] | 0.143 |
| insertion | 1 | 27/100 | 27.0% [19.3%, 36.4%] | 0.763 |
| insertion | 3 | 0/100 | 0.0% [0.0%, 3.7%] | 0.525 |
| insertion | 5 | 0/100 | 0.0% [0.0%, 3.7%] | 0.355 |
| insertion | 10 | 0/100 | 0.0% [0.0%, 3.7%] | 0.140 |
| exact copy | 0 | 200/200 | 100.0% [98.1%, 100.0%] | 1.000 |

Base: 2000 clean items from `fancyzhx/ag_news` split `test`. Planted: 200 exact copies, 1200 near-duplicate copies (100 per edit bucket), 600 clean controls. Reporting threshold 0.8 on exact Jaccard, 5-word shingles.

### E1 detector recall conditional on true Jaccard >= 0.8 (LSH recall)

| Edit | Size (words) | Planted near pairs | With true Jaccard >= threshold | Found / those pairs | LSH recall [95% CI] |
|---|---|---|---|---|---|
| all edits | all | 1200 | 88 | 85/88 | 96.6% [90.5%, 98.8%] |
| all edits | 1 | 300 | 87 | 84/87 | 96.6% [90.3%, 98.8%] |
| all edits | 3 | 300 | 1 | 1/1 | 100.0% [20.7%, 100.0%] (low n) |
| all edits | 5 | 300 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| all edits | 10 | 300 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| substitution | all | 400 | 23 | 22/23 | 95.7% [79.0%, 99.2%] |
| substitution | 1 | 100 | 23 | 22/23 | 95.7% [79.0%, 99.2%] |
| substitution | 3 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| substitution | 5 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| substitution | 10 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| deletion | all | 400 | 37 | 36/37 | 97.3% [86.2%, 99.5%] |
| deletion | 1 | 100 | 36 | 35/36 | 97.2% [85.8%, 99.5%] |
| deletion | 3 | 100 | 1 | 1/1 | 100.0% [20.7%, 100.0%] (low n) |
| deletion | 5 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| deletion | 10 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| insertion | all | 400 | 28 | 27/28 | 96.4% [82.3%, 99.4%] |
| insertion | 1 | 100 | 28 | 27/28 | 96.4% [82.3%, 99.4%] |
| insertion | 3 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| insertion | 5 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |
| insertion | 10 | 100 | 0 | 0/0 | n/a: no planted pairs at or above the threshold |

Only pairs whose exact Jaccard is at or above 0.8 can be reported, so this is the recall of the retrieval step among pairs the detector is allowed to report. Pairs below 0.8 are not counted here; the detector cannot report them by design. Above-threshold pairs missed by retrieval: 3. Below-threshold pairs reported: 0 (must be 0).

### E1 recall by true Jaccard bin

| True Jaccard | Planted near pairs | Found | Recall [95% CI] |
|---|---|---|---|
| [0.0, 0.5) | 700 | 0 | 0.0% [0.0%, 0.5%] |
| [0.5, 0.6) | 146 | 0 | 0.0% [0.0%, 2.6%] |
| [0.6, 0.7) | 81 | 0 | 0.0% [0.0%, 4.5%] |
| [0.7, 0.8) | 185 | 0 | 0.0% [0.0%, 2.0%] |
| [0.8, 0.9) | 63 | 60 | 95.2% [86.9%, 98.4%] |
| [0.9, 1.0] | 25 | 25 | 100.0% [86.7%, 100.0%] |

True Jaccard is the exact Jaccard of the planted pair's 5-word shingle sets, computed without the detector. Bins do not depend on the threshold.

### Why short texts with word edits fall below the threshold (arithmetic, not measured)

| Word edits k | Length that guarantees the threshold (words) | Jaccard lower bound at 35 words | Jaccard lower bound at 50 words | Jaccard lower bound at 100 words |
|---|---|---|---|---|
| 1 | 49 | 0.722 | 0.804 | 0.901 |
| 3 | 139 | 0.348 | 0.508 | 0.730 |
| 5 | 229 | 0.107 | 0.296 | 0.587 |
| 10 | 454 | 0.000 | 0.000 | 0.315 |

With 5-word shingles, a text of n words has S = n - 4 shingles. One word edit changes at most five of them, so Jaccard is at least (S - 5k) / (S + 5k). For a threshold of 0.8 this needs S >= 45k, that is n >= 45k + 4 words. Most base items in this run fall in the short and medium length bins (base counts in the length table), so a single word edit can take a typical item below the threshold even when the detector works. Shingle size and threshold are the knobs; see docs/limitations.md.

### E1 clean false positives

| Clean pairs | Flagged | Rate [95% CI] |
|---|---|---|
| 10000 | 0 | 0.00% [0.00%, 0.04%] |

Random pairs of clean controls; flagged at threshold 0.8.

### E1 recall by text length (found / planted)

| Edit | Size | short, under 34 words (base n 607) | medium, 34 to 40 words (base n 610) | long, 41 words or more (base n 783) |
|---|---|---|---|---|
| substitution | 1 | 4/30 = 13.3% | 3/34 = 8.8% | 15/36 = 41.7% |
| substitution | 3 | 0/30 = 0.0% | 0/30 = 0.0% | 0/40 = 0.0% |
| substitution | 5 | 0/22 = 0.0% | 0/32 = 0.0% | 0/46 = 0.0% |
| substitution | 10 | 0/23 = 0.0% | 0/30 = 0.0% | 0/47 = 0.0% |
| deletion | 1 | 5/25 = 20.0% | 6/31 = 19.4% | 24/44 = 54.5% |
| deletion | 3 | 0/30 = 0.0% | 0/33 = 0.0% | 1/37 = 2.7% |
| deletion | 5 | 0/29 = 0.0% | 0/29 = 0.0% | 0/42 = 0.0% |
| deletion | 10 | 0/35 = 0.0% | 0/23 = 0.0% | 0/42 = 0.0% |
| insertion | 1 | 2/31 = 6.5% | 1/41 = 2.4% | 24/28 = 85.7% |
| insertion | 3 | 0/33 = 0.0% | 0/32 = 0.0% | 0/35 = 0.0% |
| insertion | 5 | 0/36 = 0.0% | 0/30 = 0.0% | 0/34 = 0.0% |
| insertion | 10 | 0/32 = 0.0% | 0/31 = 0.0% | 0/37 = 0.0% |

Cells with fewer than 20 planted pairs are marked low n. Bins are word-count tertiles of the eligible pool. Paraphrase detection is out of scope: the tool has no language or embedding model.
<!-- results:end -->

## Findings on public benchmarks (E2 and E3)

<!-- results:start -->
_No results yet. The experiments have not been run in full: `make results`, then `make report`._
<!-- results:end -->

Literature values in E2 are cited as reported; they are not our measurements. Each rate in E3 is a count of matched items, not a claim about the benchmark's validity.

## Limitations

- Near-duplicate detection is lexical. Paraphrases, translations and semantic duplicates are missed. There is no language model or embedding model in this tool.
- LSH recall is probabilistic. Reported pairs are exact, but a pair just above the threshold can be missed. E1 measures how often, by edit type, edit size and text length, and separately conditional on the true Jaccard being at or above the threshold.
- Short texts with word edits fall below a 0.8 threshold on five-word shingles. This is arithmetic (see the calibration section and `docs/limitations.md`), not a detector fault.
- Shortcut flags depend on the baseline classifier. A different classifier can give a different result, so a flag is a prompt, not a proof.
- Template-driven text (for example GSM8K phrasing or ARC stems) can match without contamination.
- Membership inference against closed models is not included. Per-item evidence from model outputs is weak, so the tool does not claim it.

## What this does not show

- A clean report does not show that a benchmark is clean. It shows that these checks did not fire at the stated thresholds.
- It does not measure model memorisation, and it does not check that a score was computed correctly.
- The E1 planted copies are synthetic edits of real text. They measure the detector, not the frequency of leaks in any benchmark.
- Results apply only to the datasets in `specs/`, as the Hugging Face Hub served them when they were run.
- There is no web interface beyond the static reports.

## Datasets and licences

The licences below are as stated on each dataset card when checked on 2026-11-19. benchlint does not redistribute any dataset text; it reads the Hub at run time.

| Dataset (Hub id) | Used for | Licence |
|---|---|---|
| `openai/gsm8k` (main) | E3 | MIT (dataset card) |
| `allenai/ai2_arc` (ARC-Easy) | E3 | CC-BY-SA-4.0 (Hub tag; the card does not state it) |
| `fancyzhx/ag_news` | E1 base, E3 | Not stated on the card (Hub tag "unknown"); original release by Zhang et al. (2015) |
| `stanfordnlp/imdb` (plain_text) | E3 | Not stated on the card (Hub tag "other"); original release by Maas et al. (2011) |
| `stanfordnlp/snli` (plain_text) | E2, E3 | CC-BY-SA-4.0 (dataset card; includes Flickr30k content) |
| `nyu-mll/glue` (mnli) | E2 | The GLUE card defers to the source corpora (MultiNLI, per genre); Hub tag "other" |
| `google-research-datasets/mbpp` (full) | E3 | CC-BY-4.0 (dataset card) |
| `openai/openai_humaneval` | E3 (reference) | MIT (dataset card) |

The code is MIT (see `LICENSE`).

## Environment

Developed on macOS x86_64 (Intel i7-4770HQ laptop, CPU only). CI runs on Linux because Intel macOS runners are being retired. Every result directory has an `env.json` from `scripts/envinfo.py` (git SHA, UTC time, CPU, RAM, OS, Python and package versions; no hostname or username).

## Reproducing the experiments

```bash
make setup            # uv venv -p 3.11 .venv && uv sync --extra dev
make lint type test   # the tests run offline
make results          # the full queue: E1, E2, E3 (network and CPU heavy; not run in CI)
make report           # render results into this README
make check-results    # CI gate: fails if the README is out of date
make site             # static GitHub Pages site in site/dist
```

The experiment scripts take `--smoke` for a small check that finishes in about a minute; smoke output goes to a temporary folder and is never rendered. The settings are in `docs/prereg.md`, and every change to them is in `docs/deviations.md`.

## Development

See `AGENTS.md` for layout, invariants and conventions. `schemas/audit_report.schema.json` is generated from the `AuditReport` model in `src/benchlint/schema.py`.

## How this was built

This project was built with AI coding assistants (Claude Code, with models from the Claude family). The design, the pre-registered experiments and the final review are the author's. Each design choice is recorded with its reason in `docs/design.md` and `docs/deviations.md`. Results are generated from committed JSON, not typed by hand.

## Citation

See `CITATION.cff`.
