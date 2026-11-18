# benchlint

A linter for ML benchmarks: it finds train/test leakage, near-duplicates, label conflicts and shortcut features before you trust a score.

> **Status: scaffold (v0.1.0 in progress).** The CLI stubs exist; the checks are not implemented yet. No results have been measured or published.

## 1. Calibration: how well the detectors work (E1, placed before findings on purpose)

Before any finding, this table says how often the detectors catch planted leaks and how often they fire on clean data. A finding is only as good as this table.

<!-- results:start -->
_Not generated yet. Run `make report` after experiment E1 has completed and its result JSON is committed._

| Edit type | Edit size (words) | Planted pairs | Recall | 95% CI | Notes |
|---|---|---|---|---|---|
| substitution | 1 / 3 / 5 / 10 | TODO | TODO | TODO | |
| deletion | 1 / 3 / 5 / 10 | TODO | TODO | TODO | |
| insertion | 1 / 3 / 5 / 10 | TODO | TODO | TODO | |
| false positives | clean random pairs | TODO | n/a | n/a | false-positive rate on 10,000 clean pairs |

Out of scope: paraphrase detection. There is no LLM or embedding model in this tool.
<!-- results:end -->

## 2. Findings on public benchmarks (E2 to E4)

<!-- results:start -->
_Not generated yet. Findings appear here, generated from `results/`, after the experiments run._

Planned: hypothesis-only replication on SNLI and MNLI, leakage scans on GSM8K, MBPP vs HumanEval, ARC-Easy, AG News and IMDB, and dogfooding on the sibling projects' eval sets.
<!-- results:end -->

## What it checks

- **Exact duplicates** within and across splits (blake2b over normalised fields).
- **Near-duplicates** with MinHash LSH (5-word shingles, 128 permutations, threshold 0.8), verified by exact Jaccard.
- **Decontamination**: 13-gram overlap between eval items and a reference corpus.
- **Label conflicts** among duplicate or near-duplicate items.
- **Shortcut features**: partial-input baselines (single field, length, one numeric feature) compared with majority and chance.
- **Canaries**: insert a GUID canary into a dataset and grep corpora for it later.

## Quickstart (planned)

```bash
uv tool install git+https://github.com/aayushkrm/benchlint
benchlint demo                       # synthetic data with planted leaks, under 30 s, no downloads
benchlint audit specs/gsm8k.yaml --out reports/ --fail-on high
```

Exit codes: 0 pass, 1 check failed (a finding at or above `--fail-on`), 2 usage or runtime error.

Pre-commit hook (`.pre-commit-hooks.yaml`, hook id `benchlint-audit`) and composite GitHub Action (`action.yml`) are included for use in other repositories.

## Method limits (read before the findings)

- Exact and n-gram overlap are reliable for the leaks they target. Per-item membership inference on closed models is not reliable, so it is not included.
- Near-duplicate detection uses lexical Jaccard. Paraphrases are missed.
- Shortcut flags compare against majority and chance with a margin. A flag is a prompt to investigate, not proof.
- Template-driven datasets (for example GSM8K phrasing) can give false positives. Jaccard distributions are reported, not only counts.

## What this does not show

- It does not show that a benchmark is clean when no finding appears. It shows only that these checks did not fire at the stated thresholds.
- It does not measure model memorisation.
- It has no web UI beyond the static report.
- Results are for the datasets listed in `specs/`, under the versions the Hugging Face Hub served when they were run.

## Datasets and licences

Each dataset spec is in `specs/`. Dataset licences are listed here once they are verified at build time. `# TODO verify` marks the specs that are not yet checked against the Hub.

- GSM8K (`openai/gsm8k`): MIT. TODO verify
- MBPP (`google-research-datasets/mbpp`): CC-BY-4.0. TODO verify
- HumanEval (`openai/openai_humaneval`): MIT. TODO verify
- ARC (`allenai/ai2_arc`): CC-BY-SA-4.0. TODO verify
- AG News (`fancyzhx/ag_news`): TODO verify
- IMDB (`stanfordnlp/imdb`): TODO verify
- SNLI (`stanfordnlp/snli`): CC-BY-SA-4.0. TODO verify
- MNLI (`nyu-mll/glue`, config `mnli`): TODO verify

The code is MIT (see `LICENSE`).

## Environment

Developed and measured on macOS x86_64 (Intel i7-4770HQ laptop, CPU only). CI runs on Linux because Intel macOS runners are being retired. Environment records are written with `scripts/envinfo.py`.

## Development

```bash
make setup      # uv venv -p 3.11 .venv && uv sync --extra dev
make lint type test
```

See `AGENTS.md` for layout, invariants and conventions.

## How this was built

This project was built with AI coding assistants (Claude Code). The design, the pre-registered experiments and the final review are the author's. The author reads the code and is responsible for each design choice. Results are generated from committed JSON, not typed by hand.
