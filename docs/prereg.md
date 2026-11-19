# Pre-registration: E1 to E4

Status: pre-registered on 2026-11-19, before any full run. This commit precedes every file
under `results/`. Every change of n, threshold, cap, seed, model or method after this point is
logged in `docs/deviations.md` with a reason. No expected numbers are written here. Smoke runs
(`--smoke`) are checks of the code and are never reported.

Common settings (all experiments):

- Detector: `benchlint` as committed at the commit that adds this file (schema, io, near, labels,
  exact, ngram and shortcut modules). Near-duplicate settings: word 5-gram shingles, 128
  permutations, reporting threshold 0.8 (exact Jaccard), LSH retrieval threshold 0.7, MinHash
  seed 0. Text normalisation: `benchlint.normalize` (NFKC, casefold, punctuation deleted, digits
  kept, whitespace collapsed; applied per field).
- Seeds: every random draw uses an explicit seed, stated below. Bootstrap CIs use 1000
  resamples, seed 0. Permutation p-values use 200 permutations, seed 0.
- Uncertainty: every rate is reported with its count, its denominator and a 95% Wilson interval.
- Data source: the Hugging Face Hub. Dataset names, configs, splits and columns were checked
  against the live Hub on 2026-11-19 (see the header comments in `specs/*.yaml`).
- Environment: each result directory has an `env.json` written by `scripts/envinfo.py`.
- Full runs are executed once, in the order `make results` gives. Any crash is fixed, logged in
  `docs/deviations.md`, and the run repeated with the same settings.

## E1. Detector calibration (planted duplicates and near-duplicates)

Question: how often does the detector recover planted copies, by edit type, edit size and text
length, and how often does it fire on clean pairs?

Base data: `fancyzhx/ag_news`, split `test` (7,600 items). Only items with at least 12
whitespace-separated words are eligible (so that a 10-word edit can be applied).

Clean base set (seed 0): 2,000 items drawn without replacement from the eligible pool, then
sorted. Each base item gets one role, by a seeded permutation (seed 0) of the 2,000:

- 200 items receive an exact copy (10% of the base set).
- 1,200 items receive one near-duplicate copy (60%). The 12 edit buckets (substitution,
  deletion, insertion crossed with 1, 3, 5 and 10 words) receive 100 copies each, assigned
  round-robin along the permutation.
- 600 items receive no copy. They are the clean controls.

Planted copies are written to split `copy`. Originals stay in split `orig`. Record ids are
`orig-<base index>` and `copy-<base index>`.

Edit procedure (seed 0, per copy, applied to the whitespace tokens of the original):

- Substitution: k distinct token positions are replaced by k distinct words.
- Deletion: k distinct token positions are removed.
- Insertion: k distinct gap positions (of len+1 gaps) receive k distinct words.
- Inserted or substituted words come from a seeded (seed 0) vocabulary of alphabetic lowercase
  words from the eligible pool, excluding every word of the original text.

The word-level edit distance is k by construction. Exact copies have no edits.

Length bins, from the normalised word count of the original: short (fewer than 30 words),
medium (30 to 59 words), long (60 words or more). Bin counts are reported.

Detection: `find_near_duplicates` on the records of the `orig` and `copy` splits, with the
settings above. A planted pair counts as found when the cross-split finding lists the pair
(`details["pairs"]`) with exact Jaccard at or above 0.8. The exact copies are also checked with
the exact-duplicate check.

Reported outputs:

- Recall per (edit type, edit size), per (edit type, edit size, length bin) and per length bin,
  each with n, found count and Wilson 95% CI.
- Median exact Jaccard of the planted pairs per bucket. It is computed directly from the shingle
  sets, not read from the detector, so it is available for missed pairs too.
- Recall of the exact copies (the exact-duplicate check).

Clean false-positive rate (seed 1): 10,000 distinct unordered pairs, each drawn from the 600
clean controls. No planted copy is involved. Each pair is passed through the detector as two
records (splits `a` and `b`). A pair counts as a false positive when the detector flags it at
threshold 0.8. The Wilson interval is reported, and so is the histogram of exact Jaccard for the
10,000 pairs.

Precision of the claims: E1 measures lexical near-duplicates with word edits on news text. It
does not measure paraphrase detection, which is out of scope because the tool has no language
model or embedding model.

Power: about 100 planted pairs per (edit type, edit size). The per (edit, size, length) cells
are smaller, so their Wilson intervals are wide. A cell with fewer than 20 planted pairs is
marked "low n" in the output.

Stop rules:

- The thresholds, shingle size, permutation count and LSH setting are not changed after the full
  run. If a bucket looks poor, it is reported as it is. A follow-up with different settings is a
  new pre-registration (E1b), not an edit of this one.
- An exact copy that is not found is a bug, not a result. It is fixed, logged, and the run is
  repeated.

Script: `scripts/calibrate.py`. Output: `results/e1/calibration.json`, `results/e1/env.json`.

## E2. Known-artifact replication (hypothesis-only baseline)

Question: how well does a hypothesis-only classifier do on SNLI and MNLI, compared with the
values reported in the literature?

Literature values, cited and not claimed: Gururangan et al. (2018), "Annotation Artifacts in
Natural Language Inference Data" (NAACL). Their hypothesis-only models reach about 67% on SNLI
and about 53% on MNLI matched dev. These are reported values. The exact table cells must be
checked against the paper before the README cites them; the numbers appear in the output as
"reported", never as our measurements.

Data:

- SNLI (`stanfordnlp/snli`, `plain_text`). Train: items with label -1 removed (no gold label),
  then a seeded (seed 0) sample of at most 100,000 rows. Test: the test split with label -1
  removed (9,824 items expected; the exact count is recorded in the output).
- MNLI (`nyu-mll/glue`, config `mnli`). Train: a seeded sample of at most 100,000 rows of the
  train split. Eval: `validation_matched` (9,815 items).

Model: benchlint's `hypothesis_only` view. TF-IDF with 1 and 2-grams on the hypothesis, then
logistic regression (max_iter 1000, random_state 0). The majority baseline is the most frequent
train label. Chance is 1/3 (three classes). The shortcut cap is raised to 100,000 for this
experiment (see the deviation logged for it), so the whole capped train set is used.

Reported outputs: accuracy with bootstrap 95% CI, majority and chance baselines, permutation
p-value, and the gap in points against the reported literature value.

Stop rule: if the gap exceeds 10 points, investigate once (label encoding, the -1 filter,
tokenisation) and report both the initial run and the investigated run. The classifier is not
tuned to close the gap. A failure to match the literature is still a result. The comparison
tests a similar shortcut. It is not a replication of the same model, since the literature used
different hypothesis-only classifiers.

Script: `scripts/run_e2.py`. Output: `results/e2/snli.json`, `results/e2/mnli.json`,
`results/e2/env.json`.

## E3. Leakage scan of six public benchmarks

Question: what do the exact, near-duplicate, label-conflict, n-gram and shortcut checks report
on these train/test pairs?

Datasets and settings are the committed specs, unchanged for the run (`specs/`):

| Spec | Source | Split pair | Checks |
|---|---|---|---|
| gsm8k | `openai/gsm8k` (main) | train vs test | exact, near |
| arc-easy | `allenai/ai2_arc` (ARC-Easy) | train vs test | exact, near, labels |
| ag-news | `fancyzhx/ag_news` | train vs test | exact, near, labels |
| imdb | `stanfordnlp/imdb` (plain_text) | train vs test | exact, near, labels |
| snli | `stanfordnlp/snli` (plain_text) | train (capped 100,000, label -1 removed) vs test | exact, near, labels, shortcut |
| mbpp-vs-humaneval | `google-research-datasets/mbpp` (full) | train vs test, plus a 13-gram check of the MBPP test split against the HumanEval prompts | exact, near, n-gram |

Settings: exact keys on normalised fields; near at threshold 0.8 with 5-gram shingles and 128
permutations; 13-gram overlap for the n-gram check; seed 0 for every dataset. The shortcut check
runs with its committed views and caps.

Reported outputs, for each dataset: each finding's count, denominator, rate and 95% Wilson
interval, and up to 5 example pairs with their texts cut to 200 characters. Severity is taken
from the code (the rules are in the module docstrings). Every dataset's report is kept as
`results/e3/<spec>.json`, and `results/e3/summary.json` lists them with their mode.

Stop rules:

- Results are reported whatever they are. No threshold is changed after counts are seen.
- A dataset that fails to load is recorded in the summary with its error message. It is not
  dropped.
- A "leak" here means the lexical check matched. Shared templates (for example ARC stems or
  GSM8K phrasing) can produce matches that are not contamination. The report says so and gives
  the example pairs for a reader to check.

Script: `scripts/run_e3.py`. Output: `results/e3/*.json`, `results/e3/env.json`.

## E4. Dogfooding

Planned: run benchlint over the ARC and GSM8K evaluation subsets of the sibling project
`quantgate` against their train splits, and over the task instructions of the sibling project
`passk` (near-duplicate check). E4 is written once the sister repositories exist (since run; see the E4 settings amendment below).
Its settings, sizes and results are not pre-stated here. The same rules apply: report whatever
comes out, with n and CI, and log any change in `docs/deviations.md`.

## Thresholds, n and decision rules (summary)

- Near-duplicate reporting threshold: 0.8 (exact Jaccard). LSH retrieval threshold: 0.7.
- Caps: SNLI and MNLI train at 100,000 rows (seed 0). Shortcut cap 100,000 for E2 and E3 SNLI
  and MNLI. Other splits are uncapped.
- Power: E1 has about 100 planted pairs per edit bucket and 10,000 clean pairs. E2 has about
  9,800 test items per SNLI and MNLI evaluation. E3 has all items of each split (within caps).
- Literature gap threshold for investigation in E2: 10 points.

## Amendment (2026-11-19, before any full run)

Recorded in `docs/deviations.md`, with the reasons:

- Near and label tokens: shingles are built from `benchlint.normalize.word_tokens` (punctuation
  becomes a space), not from the punctuation-deleting `normalize_text`. The exact key still uses
  `normalize_text`.
- E1 length bins are word-count tertiles of the eligible pool (short under 34 words, medium 34 to
  40, long 41 or more), not 30 and 60.
- E1 false-positive pairs are drawn from the 600 clean controls.
- The shortcut check's training cap for SNLI and MNLI is 100,000 rows.
- Wording only: the E2 stop rule refers to the initial run (banned-word rule, AGENTS.md invariant 9).
- Literature check (2026-11-19, after the first E2 run): SNLI test 67.0 and MultiNLI matched 53.9 are verified against Table 2 of Gururangan et al. (2018), arXiv:1803.02324. The check required by the E2 section is satisfied; the MNLI constant in the first run (53) is corrected, see docs/deviations.md.

## E4 settings (amendment, 2026-11-19, committed before any E4 run)

The E4 section above leaves the settings open. This amendment fixes them before the run. It is
the only E4 text that the run follows; any later change goes in `docs/deviations.md`.

**Scope.** E4 is dogfooding: benchlint applied to two sibling projects' inputs. Its n is small
(300, 100 and 24 items), so its intervals are wide and it is not a benchmark result. Its question
is whether the sibling inputs contain duplicates, not how a model performs.

**Sibling inputs (read from the sibling checkouts, never copied with absolute paths).**

- `quantgate`: `data/subsets/arc_easy_test.json` (300 committed IDs) and
  `data/subsets/gsm8k_test.json` (100 committed IDs). The text is loaded from Hugging Face:
  ARC-Easy (`allenai/ai2_arc`, config ARC-Easy) by its `id` column, and GSM8K (`openai/gsm8k`,
  config main) by its 0-based row index in the test split. Every committed ID must be found, or
  the run fails. No ID is dropped silently.
- `passk`: the 24 task files `src/passk/tasks/labstock/*.yaml`, field `instruction` only. The
  `initial_state` lists and other fields are not checked.

**Text.** Eval and train items are compared on the text that quantgate's prompt contains: ARC
question plus its choice texts, joined by spaces in the dataset's order; GSM8K question only (the
gold answer is not in the prompt). This is a deviation from the E3 ARC spec, which uses the
question alone. It is logged in `docs/deviations.md`.

**Checks and thresholds (the same as E3, not tuned).** Exact duplicates on the normalised key;
near duplicates with word 5-gram shingles, 128 MinHash permutations, seed 0, reporting threshold
0.8 on exact Jaccard and LSH retrieval at 0.7. Labels, shortcut and n-gram checks are not run:
they do not answer the dogfooding question.

**Quantgate direction and metric.** The reference is the train split (ARC-Easy 2,251 items;
GSM8K main 7,473 items). The measured items are the committed eval subset. The metric is the
number of eval items with at least one train partner, out of the eval subset size, with a Wilson
95% interval. Exact matches are a subset of near matches, because identical text has identical
shingles. The E3 metric counted items on both sides, so the two are not comparable. Up to five
example pairs per check are kept, sorted by similarity, each text cut to 200 characters. The
same excerpt rule as E3 applies.

**Passk.** Within the 24 instructions: exact duplicates (count of repeated instructions), near
duplicates at 0.8 (count of instructions with a partner), and all 276 pairwise exact 5-gram
Jaccard values, summarised by minimum, median, maximum, histogram and the count at or above 0.8.
No threshold is chosen for "templated". The Jaccard distribution is the description. Instructions
are 3 to 25 words, so a 5-gram shingle is a large part of a short instruction. A near-duplicate
count of zero is expected and is reported as a null result, not as evidence that the tasks are
distinct.

**Provenance.** Each result records, for each sibling: its git commit, whether its working tree
has uncommitted changes, and the changed paths. It also records a SHA-256 of each input file
(committed subsets, task files) and the item counts. No absolute path is written to any result.

**Stop rules.** The thresholds above are not changed after the run. A missing ID, a failed load or
a missing sibling checkout fails the run with an error (recorded, not dropped). A null result is
reported in the README and the headline as a null result.

