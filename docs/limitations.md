# Limitations

What each check can and cannot show. Read before quoting a finding.

## Matching is lexical

- Exact keys, word shingles and 13-grams match surface text. Paraphrases, translations and
  reordered sentences are missed. E1 states the recall on edited copies, so the limit is measured
  for word-level edits only.
- The n-gram check counts items that share at least one 13-gram with the reference. A shared
  boilerplate phrase can match without contamination.

## LSH recall is probabilistic

- Reported pairs are exact (verified with Jaccard). A pair that is truly above 0.8 can still be
  missed by LSH retrieval. The retrieval threshold is 0.7, which lowers the miss rate; it does not
  remove it. E1 gives the measured recall for the planted edits, by edit type, size and text
  length.

## Thresholds are conventions

- 0.8 is the reporting threshold and 5-word shingles are the default. Other choices change the
  result. Both are configurable, and E1 reports how the default behaves.
- A near-duplicate count depends on the shingle size: short texts have fewer shingles, so one
  edit moves Jaccard more. E1 bins by text length for this reason.

## Shortcut baselines

- A flag depends on the baseline classifier (TF-IDF with logistic regression, a shallow decision
  tree for length, a single numeric feature). A stronger classifier could find more signal. The
  flag says the chosen view beats majority by a margin, not that the field has no other signal.
- Permutation p-values for the numeric scan are not corrected for multiple comparisons.

## Templates and false positives

- Templated text (GSM8K phrasing, ARC stems) shares many shingles. Such pairs can be flagged.
  Each near finding reports its Jaccard histogram and its example pairs so a reader can check.

## Membership inference is out of scope

- Telling whether a closed model memorised an item needs model outputs, and those are weak
  evidence. benchlint does not claim to detect memorisation.

## Data and licences

- Results depend on the dataset versions the Hub served at run time. The spec files name the
  configs and splits checked on 2026-11-19; the Hub can change.
- Some dataset licences are not stated on their cards (see README). benchlint does not
  redistribute the text.

## Experiments

- E1 uses synthetic edits of real text. Its results measure the detector on known edits and say
  nothing about how often benchmarks leak.
- E2 compares with literature values that are cited as reported. The classifier differs from the
  one in the cited paper, so the comparison is about shortcut strength, not a replication.
- No full results are committed yet. Until `make results` runs and its output is committed, the
  README and site say that no results exist.
