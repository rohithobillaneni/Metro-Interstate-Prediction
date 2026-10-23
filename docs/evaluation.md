# Chronological evaluation protocol

This protocol was fixed before executing the new model comparison. The question
is whether an existing-feature model improves on a training-only weekday/hour
baseline for **2017 same-hour traffic estimation using observed weather**.
Observed weather is not assumed available at an earlier forecast origin. This
is not evidence of advance forecasting quality.

## Data and cohort

Use the raw CSV identified in [DATASET.md](../DATASET.md), never the historical
processed/test CSVs. Parse timestamps as naive local clock times exactly as
provided; do not infer daylight-saving transitions or a timezone conversion.
Reject malformed/nonfinite numeric values, negative traffic, inconsistent
targets at the same timestamp, invalid cloud percentages and missing fields.
Keep zero traffic. Exclude nonpositive Kelvin temperatures by a fixed physical
rule, not a quantile learned from any partition. No IQR filtering is applied.

Several rows can describe the same hour with different weather labels. Check
target agreement across every row, then retain the first source-order row with
a valid temperature for each timestamp. This is deterministic for a fixed CSV,
but weather choice is order-dependent and loses other weather descriptions.
Deduplication and fixed cutoffs ensure no timestamp appears in two partitions.
The report accounts for every raw row as retained, invalid-temperature excluded
or duplicate extra. The known raw dataset also has substantial time gaps; it
is not resampled or imputed as a regular hourly series.

## Fixed partitions and models

| Partition | Calendar interval | Role |
|---|---|---|
| Train | Before 2017-01-01 | Fit all estimators and feature transformations |
| Validation | 2017-01-01 inclusive to 2018-01-01 exclusive | Compare all three prespecified models |
| Reserved test | From 2018-01-01 | Counted/validated only; no fitting or scoring |

No hyperparameter search or automatic winner selection is performed.

- Training mean: mean traffic across training timestamps.
- Weekday/hour mean: training mean for each weekday and clock hour, falling
  back to the training global mean for unseen combinations. Report fallback count.
- Random forest: 100 trees, max_depth 12, min_samples_leaf 2, random_state 42,
  n_jobs 1. Reuse the original feature pipeline: Celsius temperature/clouds,
  weather category, holiday indicator, hour category and cyclical calendar terms.
  The CSV string `None` is a non-holiday; named holidays are holidays.

All models predict exactly the same validation timestamps. Record MAE, RMSE,
R2 and monthly breakdowns, including unfavorable results. R2 is undefined for
constant targets and represented by JSON null. Prediction digests help detect
changes within a fixed environment; identical floating-point bytes across
different platforms or dependency versions are not promised.

## Reproduction

From the repository root, with Python 3.13:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-evaluation.txt
.venv/bin/python -m unittest discover -s tests -p 'test_offline_data.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_evaluation.py' -v
.venv/bin/python -m src.evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output validation-result.json
```

The input requirements specify only the evaluation stack; the generated lock
pins transitives and verifies downloaded artifact hashes. The legacy broad
`requirements.txt` and `setup.py` are not used for this path. The command does
not overwrite old results or any historical artifact. An omitted `--output`
writes JSON to stdout. Installation needs package access; evaluation is offline.
For the complete test suite including local API/artifact tests, install
`requirements-serving.txt` and run `python -m unittest discover -s tests -v`.

## Limits and historical comparison

The original training path used a random split, dropped timestamp identity and
had no calendar baseline. Its stored MAE/R2 cannot be compared directly with
this protocol. The historical data path's row identity is not recoverable from
its processed files, so actual train/test duplicate leakage is not asserted.
The original pipeline did fit its scaler/encoders inside cross-validation;
the earlier distribution-based row filtering was performed before splitting.

The first new comparison is one fixed historical validation year on one road.
It provides no confidence interval, multi-road generalization or deployment
claim. Validation labels have now been used for development evaluation; 2018
must remain reserved until a final protocol is frozen. A later protocol should
include rolling-origin validation and exact-time seasonal lag baselines with
explicit missing-hour/feature-availability rules. Do not use row shifts as a
substitute for elapsed time in this irregular series.
