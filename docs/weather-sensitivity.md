# Weather-row sensitivity across 2017

This extends the four-week [development evaluation](temporal-evaluation.md).
It is a paired preprocessing sensitivity experiment, with a frozen forest and
three unchanged baselines. It does not select a new weather-row policy or score
2018. The [machine-readable protocol](weather-sensitivity-protocol.json) was
saved locally before the full-data experiment; the evaluator rejects protocol
drift. This was a local predeclaration, not an independently preregistered study.

## Design

There are 53 nonoverlapping expanding-origin folds: consecutive seven-day
windows starting 2017-01-01, followed by a one-day 2017-12-31 window. Each
retained 2017 timestamp is scored once. Training includes only observations
strictly before the fold origin, including earlier development windows. Empty
windows fail instead of changing the dates. A fresh preprocessing pipeline and
forest are fitted in every fold and policy, with the existing fixed parameters.

The source can contain multiple weather rows for the same timestamp and target.
Compare the first and last valid-temperature rows in source order, both during
training and validation. Require target agreement even in rows subsequently
excluded for nonpositive Kelvin temperatures. Require identical timestamp/target
cohorts and cleaning counts for both policies. Report how often each feature
changes. This is source-order sensitivity: neither alternative is an exhaustive
bound over all possible ways to represent multiple weather descriptions.
The default loader and historical commands continue to select the first row.

The global mean, weekday/hour mean and exact seven-calendar-day lag baselines
use only training targets. Missing lag timestamps use the training weekday/hour
mean, then global mean. Their predictions must be identical between policies.
Seven-day maximum windows keep all exact lag references before the origin.
No gap filling, nearest-row matching, timezone inference or future labels.

Every prediction is retained with its timestamp and target. MAE and RMSE are
computed from individual residuals, pooled across observations rather than
averaged across folds. Fixed calendar-quarter results describe temporal
variation. Forest-minus-baseline MAE and last-minus-first MAE/RMSE use identical
cohorts. Negative sensitivity differences favor the last-row policy on this
experiment only; they do not trigger policy selection or tuning.

## Reproduce

Use the locked Python 3.13 source-checkout environment in
[CONTRIBUTING](../CONTRIBUTING.md). No new dependencies, services or model files:

```sh
python -m unittest discover -s tests -v
python -m src.weather_sensitivity --data data/Metro_Interstate_Traffic_Volume.csv --output weather-result.json
```

The output path must be new; omitted output prints JSON. A full run fits 106
forests on one worker each. The report includes protocol, dataset, implementation
and dependency-lock hashes, environment, coverage, fallback counts, every
prediction and all comparisons. CI reproduces the complete protocol and cohorts
on Linux. Exact floating-point agreement across platforms is not promised.
Changes to the implementation produce different code hashes; historical reports
retain the hashes of their original implementation and are never overwritten.

## Limits

2017 is development data, already inspected in earlier work. This is retrospective
same-hour estimation with observed weather, not advance forecasting. Feature and
label availability latency is unknown. Missing clock slots are not ground truth;
naive local clocks do not establish elapsed UTC hours across DST. Report all
quarters, folds and both policies; do not infer significance or choose favorable
subsets. A different full-year score from the old four-week score is not evidence
of model improvement. 2018 targets/features remain excluded before value
validation, fitting and scoring. Source shape/timestamps are still parsed.

Derived benchmark predictions use the UCI dataset under CC BY4.0; see
[DATASET.md](../DATASET.md). Historical results remain unchanged.

## Recorded development results

[Raw report](../benchmarks/weather-sensitivity-2017.json), produced on macOS
arm64 with Python 3.13.14 and the recorded dependency locks. The complete clean
environment repeat was byte-identical. Each execution fitted 106 forests and
took approximately 400 seconds on the local machine; this is a reproduction
observation, not a comparative performance benchmark.

There are 8,713 observed validation hours, 47 absent naive clock slots out of
8,760, and a longest absent run of 9 hours. No missing targets were filled.
The seasonal baseline found 8,667 exact prior-week targets and used the training
calendar fallback 46 times; no global fallback was needed.

First versus last selection changes features at 4,293 of 34,032 development
timestamps. Within 2017, 1,349 timestamps change: 1,348 weather categories and
one temperature, with no cloud or holiday changes. Earlier training data also
changes, so prediction differences cannot be attributed only to altered
validation features. All baseline predictions and target cohorts are identical.

| Model / weather policy | Pooled MAE (vehicles/hour) | Pooled RMSE (vehicles/hour) |
|---|---:|---:|
| training_mean / both | 1743.144 | 1985.385 |
| weekday_hour_mean / both | 272.328 | 456.332 |
| seasonal_7_day / both | 301.986 | 563.093 |
| random_forest / first | 253.586 | 434.444 |
| random_forest / last | 253.030 | 434.151 |

The paired last-minus-first differences are -0.556 MAE and -0.293 RMSE.
These aggregate differences do not establish significance or select a policy.
The existing first-row default is retained.

| Quarter | Hours | Calendar MAE / RMSE | Seasonal MAE / RMSE | First-row forest MAE / RMSE | Last-row forest MAE / RMSE |
|---|---:|---:|---:|---:|---:|
| Q1 | 2141 | 279.848 / 432.199 | 307.361 / 500.559 | 250.423 / 399.311 | 248.531 / 399.441 |
| Q2 | 2175 | 238.339 / 381.289 | 256.680 / 470.223 | 199.798 / 344.286 | 202.125 / 346.503 |
| Q3 | 2197 | 262.313 / 445.283 | 287.156 / 566.302 | 251.661 / 444.058 | 252.432 / 447.180 |
| Q4 | 2200 | 308.616 / 548.815 | 356.357 / 687.791 | 311.765 / 527.471 | 308.333 / 522.324 |

The first-row forest has higher Q4 MAE than the calendar baseline (311.765
versus 308.616), despite lower pooled MAE. Last-row forest MAE is worse than
first-row MAE in Q2 and Q3; its Q1 RMSE is also slightly higher. No uniform
policy advantage follows. All weekly predictions and quarter metrics,
including the global-mean baseline, remain in the raw report.

The prior four-week report is unchanged, including its seasonal-baseline RMSE
advantage. This experiment changes the evaluation cohort and fit origins, not
the frozen model configuration; differences from that earlier aggregate are
not a model-improvement claim. No reserved 2018 target was scored or validated
by this experiment. These results support the bounded retrospective development
evaluation, not untouched-test performance or a deployment claim.
