# Development-only temporal validation

This is the M2 experiment declared before execution. It asks whether the fixed
existing-feature forest improves on training-only mean, calendar and seasonal
baselines across four development weeks. It preserves the original
[M1 protocol and results](evaluation.md).

The task is **retrospective same-hour traffic estimation with observed weather**.
Weather and traffic publication latency are unknown. This does not demonstrate
advance forecasting, real-time availability or deployment suitability.

## Data isolation

Use the attributed [UCI raw CSV](../DATASET.md). The new
`load_development_csv` path parses timestamps and excludes all rows from
2018-01-01 onward before target/feature validation or record construction.
Header, row width and timestamp validity still apply to all rows so that they
can be routed. Changing reserved target/feature values cannot affect development
predictions, metrics or validation failures; it necessarily changes the raw-file
SHA256. Reserved raw rows are counted, never fitted or scored.

The existing M1 `load_csv` command retains its historical behavior, including
validating reserved targets before partitioning. Do not describe M1 as never
reading those labels. Neither evaluation command loads old pickle artifacts.

For retained development data, preserve zero traffic; reject nonfinite/negative
traffic and conflicting targets at the same timestamp. Drop nonpositive Kelvin
temperatures using the existing fixed rule, then retain the first valid-temperature
weather row per timestamp. Weather selection is source-order-dependent; no
aggregation or uncertainty about multiple weather descriptions is resolved here.
Report every row as reserved, invalid-temperature, duplicate extra or retained.

## Fixed folds and models

| Fold | Expanding training | Validation interval, end exclusive |
|---|---|---|
| January | Before 2017-01-01 | 2017-01-01 to 2017-01-08 |
| April | Before 2017-04-01 | 2017-04-01 to 2017-04-08 |
| July | Before 2017-07-01 | 2017-07-01 to 2017-07-08 |
| October | Before 2017-10-01 | 2017-10-01 to 2017-10-08 |

Reject empty folds; do not move cutoffs to obtain a result. Timestamps must be
unique, naive and aligned to whole clock hours. No timezone or DST inference is
made. Earlier development labels can enter later training folds. All 2017 has
already been used for development; these windows are not untouched test data.

Every model predicts the same ordered timestamps within each fold:

- `training_mean`: mean of that fold's training targets.
- `weekday_hour_mean`: training weekday/hour mean, then global mean if missing.
- `seasonal_7_day`: target at exactly seven naive calendar days earlier, from
  the training set only. A missing timestamp falls back to the training
  weekday/hour mean, then the global mean. Zero is a valid exact match. Do not
  shift rows, choose nearest neighbors or fill missing targets from validation.
- `random_forest`: a fresh preprocessing pipeline and model in each fold, using
  the same 100 trees, depth 12, minimum leaf size 2, seed 42 and one worker as M1.
  Fit all scaling/category discovery on that fold's training records only.

Seven-day validation windows ensure every possible seasonal reference precedes
its fold origin. Seven local calendar days are not asserted to be 168 UTC hours.
The lag baseline is also retrospective: the source does not establish when
traffic labels become available for an operational forecast.

## Report and reproduction

With the existing locked Python 3.13 environment, from the repository root:

```sh
python -m unittest discover -s tests -p 'test_temporal_evaluation.py' -v
python -m src.temporal_evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output temporal-result.json
```

Installation and full-suite commands are in [CONTRIBUTING](../CONTRIBUTING.md).
`--output` must name a new file; omitting it prints JSON. The report records the
protocol, source and implementation hashes, dependency lock/environment,
cleaning counts, ordered timestamp/target/prediction rows and fold metrics.

Coverage reports observed hours, expected clock slots, missing hours and the
longest missing run. Validation coverage includes leading/trailing gaps in the
fixed window. Training coverage starts at its earliest retained timestamp and
ends at the fold origin. These are naive clock-grid diagnostics, not measured
elapsed UTC durations. No gap is imputed. Seasonal exact/calendar/global counts
sum to the validation count.

MAE is in vehicles/hour; RMSE gives larger errors more weight. Compute pooled
metrics from all residuals, not unweighted averages of fold metrics. Paired MAE
differences are **forest minus baseline** on identical timestamps; negative
favors the forest on that cohort. Preserve every fold, including unfavorable
results. No automatic winner selection or tuning is performed.

The four weeks validate this bounded protocol; they do not establish annual
performance, statistical significance, multi-road generalization or superiority.
2018 remains unscored until a later final protocol is frozen. Future work may
justify more development windows, weather ablation, duplicate-row sensitivity
or dependence-aware uncertainty. Do not infer those results from this experiment.

Source changes in `offline_data.py` invalidate earlier local serving bundle
source pins by design. Retrain a fresh trusted bundle using the current source
or its matching container image; never bypass integrity checks to reuse it.

The [complete first result](../benchmarks/temporal-development-2017.md) includes
every fold and prediction. This command is supported from the source checkout
and evaluation CI environment, not inside the serving-only container image.
