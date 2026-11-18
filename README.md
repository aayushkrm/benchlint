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

<!-- results:start -->
_No results yet. The experiments have not been run in full: `make results`, then `make report`._
<!-- results:end -->

## Findings on public benchmarks (E2 and E3)

<!-- results:start -->
_No results yet. The experiments have not been run in full: `make results`, then `make report`._
<!-- results:end -->

Literature values in E2 are cited as reported; they are not our measurements. Each rate in E3 is a count of matched items, not a claim about the benchmark's validity.

## Limitations

- Near-duplicate detection is lexical. Paraphrases, translations and semantic duplicates are missed. There is no language model or embedding model in this tool.
- LSH recall is probabilistic. Reported pairs are exact, but a pair just above the threshold can be missed. E1 measures how often, by edit type, edit size and text length.
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
