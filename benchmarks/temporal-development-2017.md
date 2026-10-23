# Four-window development comparison

First execution: 24 October 2026 (Europe/London), with the fixed [M2 protocol](../docs/temporal-evaluation.md). No tuning or window changes followed these results. The complete report, including all 666 timestamped predictions for each model, is [temporal-development-2017.json](temporal-development-2017.json).

Reproduce from the repository root using Python 3.13 and the existing hash-locked dependencies:

```sh
python -m src.temporal_evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output temporal-result.json
```

The source is the attributed public UCI 492 dataset; version/hash and CC BY 4.0 terms are in [DATASET.md](../DATASET.md). Code hashes, dependency versions, Python/system/architecture and the evaluation lock hash are recorded in the report. The first run used macOS ARM64. No paid service, saved model or network call is involved in evaluation.

## Pooled results

| Model | MAE (vehicles/hour) | RMSE (vehicles/hour) |
|---|---:|---:|
| training_mean | 1642.86 | 1894.80 |
| weekday_hour_mean | 355.60 | 645.72 |
| seasonal_7_day | 355.81 | 623.41 |
| random_forest | 314.52 | 628.78 |

The forest has lower pooled MAE on these 666 observations, but the seasonal baseline has lower pooled RMSE. This trade-off is retained; the experiment does not select a winner. Metrics are calculated from pooled residuals, not averages of fold scores.

## Every fold retained

| Validation start | Train hours | Validation hours | Calendar MAE | Seasonal MAE | Forest MAE |
|---|---:|---:|---:|---:|---:|
| 2017-01-01 | 25319 | 168 | 427.62 | 436.54 | 359.80 |
| 2017-04-01 | 27460 | 166 | 236.22 | 212.73 | 182.25 |
| 2017-07-01 | 29635 | 164 | 510.83 | 552.23 | 503.40 |
| 2017-10-01 | 31832 | 168 | 250.02 | 224.74 | 215.56 |

Each window is exactly seven naive calendar days, end-exclusive. April is missing two clock slots and July four; no interpolation or moving cutoff was used. 665 seasonal predictions have an exact training lag and one October observation uses the training calendar fallback. All models share the same timestamp cohort. Full fold RMSE, coverage, fallback counts, paired MAE differences and raw predictions are preserved in JSON.

Cleaning accounts for 48,204 raw rows: 7,949 reserved rows excluded before value validation, 10 invalid-temperature rows, 6,213 duplicate extras, and 34,032 retained development timestamps. Reserved fields are read as raw CSV strings for parsing and whole-file hashing. Their values are not converted into targets/features, validated, fitted or scored by this command.

## Limits

These four weeks were already development data, not a fresh test or an estimate of annual quality. There is no significance test or confidence interval. Observed weather and prior traffic-label publication times are not established, so this remains retrospective estimation. Timestamp deduplication chooses one source-order weather description; clock times are naive and no DST resolution is inferred. Results do not establish multi-road generalization or deployment suitability.

The [original full 2017 report](validation-2017.json) remains unchanged. Its cohort and training schedule differ, so do not interpret the difference between reports as an improvement. 2018 remains unscored. Future protocol extensions must preserve this result and be defined before execution.
