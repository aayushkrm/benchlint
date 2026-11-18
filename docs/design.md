# Design

Why benchlint is built the way it is. Each decision names the alternative it rejects.

## Scope

- Lexical checks only: exact keys, word shingles, n-grams and simple partial-input baselines.
  Each one is cheap, deterministic and easy to explain.
- Rejected: embedding-based semantic deduplication (needs torch or sentence-transformers; listed
  as future work), model memorisation probes (weak evidence on closed models), image and large
  tabular datasets, and a web UI beyond static reports.

## Normalisation (two rules, on purpose)

- **Exact key:** `benchlint.normalize.normalize_text`. NFKC, casefold, every Unicode `P*`
  character deleted, digit runs kept, whitespace collapsed; each field normalised separately and
  joined with a unit separator. Deletion means "U.S." and "us" get the same key. That is the
  intended identity for exact duplicates, and the cost is accepted.
- **Word tokens:** `benchlint.normalize.word_tokens`, used by shingles, labels and n-grams. Same
  steps, but each `P*` character becomes a space, so "well-known" is two words and "don't" is
  "don t". Symbols (`S*`, for example `$` and `+`) are kept, so arithmetic text keeps its
  operators.
- Why two rules: deleting punctuation in word tokens merged distinct words ("well-known" became
  "wellknown", "U.S." became "us"), which made distinct texts look identical to the near and
  n-gram checks. The exact key keeps deletion because it is an identity test, not a word test.
- The change is in `docs/deviations.md`, with the review item that prompted it.

## Near-duplicates

- Word 5-gram shingles (the whole text when shorter than five words), 128 MinHash permutations,
  seed from the spec.
- Reporting threshold 0.8 on exact Jaccard. Every reported pair is verified with the exact
  Jaccard of its shingle sets, not the MinHash estimate.
- LSH retrieval threshold 0.7 (`NearConfig.lsh_threshold`, effective value
  `min(threshold, lsh_threshold)`). Why: datasketch tunes its bands for the threshold it is
  given, and at 0.8 a fraction of pairs just above 0.8 were not retrieved. On a synthetic probe
  of 1,216 pairs with exact Jaccard at or above 0.86, retrieval at 0.8 found 1,188 and at 0.7
  found all 1,216. The probe is synthetic and gives no guarantee for other data, so E1 measures
  recall on real text.
- Pairs are listed in `details["pairs"]` as `[a_id, b_id, jaccard]`, capped at 5,000 per finding,
  so planted-vs-found can match pairs with pairs.

## Shared helpers

`record_shingles`, `minhash_of`, `jaccard` and `lsh_candidates` in `checks/near.py` are the only
MinHash and LSH code. `checks/labels.py` uses them. Keeping one copy means one place to get
recall and verification right.

## Label handling

- Labels are compared as stripped text. A row with a blank label is treated as unlabelled.
- `DatasetSpec.drop_label_values` removes rows before the seeded cap and before every check, so
  the labels check never sees a sentinel such as SNLI's -1. A dataset with another no-label
  sentinel must list it in the spec. Without that, two copies with -1 and 0 are a conflict.

## Demo and planted-leak comparison

- `benchlint demo` plants known leaks in a synthetic dataset and prints, per category, the number
  planted and the number found. Pair categories (exact, labels, near) are matched by pair.
  Items and pairs are never compared. An n-gram row compares quoted items with matched items.
- A category without a pair listing is "not attributed", never counted.

## Exit codes and JSON

- 0 pass, 1 finding at or above `--fail-on`, 2 usage or runtime error. `--fail-on none` never
  gives 1.
- With `--json`, an error is one JSON object on stdout (`{"error", "exit_code": 2}`), the same
  shape as the sibling tools. Without it, the error goes to stderr.
- Every finding's `details` carries `rate_ci_low` and `rate_ci_high`, so machine readers get the
  same intervals as the reports.

## Experiments and results

- Every experiment has a pre-registration (`docs/prereg.md`), committed before the full run.
  Changes go in `docs/deviations.md`.
- Results are JSON under `results/<exp>/`, each with `env.json`, a `config_hash` and a `mode`.
  Only `mode: full` is rendered into the README or published on the site. Smoke output goes to a
  temporary folder.
- Resume: a stored stage is reused only when its config hash matches.
- The README and site are rendered from the JSON by `scripts/render_readme.py` and
  `scripts/build_site.py`, using the same table definitions (`benchlint.experiments`). CI runs
  `make check-results`, so the committed README cannot drift from the committed results.

## Dependencies

- scikit-learn, pandas, pyarrow, datasets and datasketch are base dependencies. The shortcut
  check, Hugging Face loading and LSH are core features, not optional extras. The review
  suggested extras; that was not adopted for this reason.
- `jsonschema` is not installed in the environment, so the schema is not validated at runtime;
  the committed schema is compared with the model in the tests instead.

## Provenance

- Each run writes `env.json` with the commit (`git_sha`), whether the code tree had uncommitted
  changes (`git_dirty`), the uncommitted paths (`git_dirty_paths`) and a hash of the uncommitted
  code diff (`git_diff_sha256`). Paths under `results/` do not count as dirty: a run writes its
  own outputs there.
- The README and `results/headline.json` take each experiment's commit and state from that
  experiment's own `env.json`; they never quote one commit for all of them.
- Runs made before these fields existed record only the commit and the dirty flag. Their
  uncommitted paths were not recorded, so `docs/deviations.md` reconstructs what can be
  reconstructed from the git history and says what cannot.

