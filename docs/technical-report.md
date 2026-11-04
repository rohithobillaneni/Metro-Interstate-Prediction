# Temporal validation and weather-row sensitivity in Metro traffic estimation

Rohith Obillaneni · 4 November 2026

This report asks how a fixed traffic-estimation model compares with meaningful
calendar and seasonal baselines when timestamps, training boundaries and weather
row choices are explicit. It synthesizes three existing development experiments;
it introduces no new model, tuning or experiment. **2018 remains unscored.**

The forest has lower pooled mean absolute error (MAE) than the baselines in all
three recorded experiments. That is not a uniform advantage: a seasonal baseline
has lower root mean squared error (RMSE) in the four-week experiment, and the
calendar baseline has lower Q4 MAE than the first-row forest in the full-year
experiment. Selecting the last weather row instead of the first changes pooled
MAE by less than one vehicle/hour, with mixed quarter-level effects. The existing
first-row default is retained.

## Question and intended use

The task is retrospective **same-hour traffic estimation using observed weather**
at one road station. It is not an advance forecast. The source does not establish
when weather observations or traffic labels become available to a deployed
system. A chronological split controls which observations enter training; it
cannot establish operational feature availability by itself.

The intended reader is an engineer auditing a small time-dependent ML project.
The contribution is the explicit evaluation contract, reproducible comparisons
and failure checks, rather than a new learning algorithm. The historical random
split and saved model results are preserved, but their timestamp identities and
full production path are not reconstructed. This report does not quantify their
leakage, compare their scores with these experiments or attribute all measured
differences to switching split strategies.

## Data and record accounting

The source is John Hogue's *Metro Interstate Traffic Volume* (2019), UCI dataset
492, [DOI 10.24432/C5X60B](https://doi.org/10.24432/C5X60B): aggregate hourly
westbound I-94 counts at Minnesota DoT station 301, with weather and holiday
attributes. The [UCI record](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
licenses the dataset under CC BY 4.0. Repository source code is separately MIT
licensed. [DATASET.md](../DATASET.md) preserves attribution, source hashes,
redistribution requirements and the historical derivative-data limitations.

The bundled CSV SHA-256 is
`0411ba6c43127a29cefa4d8f39708b9d77979d9dcf895f2d212506fe947b673c`.
The protocols preserve zero traffic, reject nonfinite/negative targets and
conflicting targets for the same timestamp, and exclude nonpositive Kelvin
temperatures by a fixed rule. They do not learn a quantile filter from the whole
dataset. Weather duplicates become one record per timestamp; the first valid
row in source order is the default. Target agreement is required even for a row
subsequently excluded for its temperature.

| Accounting | M1 fixed-year loader | M2 development loaders |
|---|---:|---:|
| Raw rows | 48,204 | 48,204 |
| Reserved raw rows excluded before value validation | Not this loader's contract | 7,949 |
| Nonpositive-temperature rows excluded | 10 | 10 |
| Duplicate extras | 7,629 | 6,213 |
| Retained timestamps | 40,565, including the reserved partition | 34,032, all before 2018 |

These are different accounting boundaries, not different dataset versions.
M1 validates reserved values before partitioning; its 6,533 reserved timestamps
are counted but never fitted or scored. M2 parses row shape and timestamps to
exclude rows from 2018 onward **before target/feature value validation**. Whole
file hashing still includes reserved bytes. Reserved-value mutation tests verify
that M2 development predictions and scores stay unchanged; the file hash changes.

Timestamps remain naive local-clock values as supplied. No timezone or DST
conversion is inferred. Among 8,760 naive clock slots in 2017, the full-year
report contains 8,713 observations, 47 absent slots and a longest absent run of
nine slots. Missing hours are not filled with estimated targets. These coverage
counts are not measurements of elapsed UTC time.

## Three evaluation contracts

The protocols were declared locally before their respective execution, not
independently preregistered. The full 2017 year was already used for development
before the later experiments. None is an untouched test-set estimate.

| Experiment | Training at each fit | Scored observations | Purpose |
|---|---|---:|---|
| M1 fixed year | Before 2017-01-01, one fit | All 8,713 retained 2017 hours | Establish a chronological reference and calendar comparison |
| M2 four weeks | All retained observations before each origin | 666 hours across four seven-day windows | Check expanding training and exact seasonal lookup |
| M2 full-year sensitivity | All retained observations before each origin and separately for each row policy | 8,713 hours per policy, each scored once | Broaden temporal coverage and compare paired first/last weather rows |

The four week origins are 1 January, 1 April, 1 July and 1 October 2017. Their
training counts are 25,319, 27,460, 29,635 and 31,832; validation counts are 168,
166, 164 and 168. The full-year protocol uses 52 seven-day windows followed by
one one-day window on 31 December. All intervals are end-exclusive. Empty or
invalid windows fail rather than moving the dates. Earlier development labels
can enter later training folds, but a fold never trains on its own validation
labels. The two full-year policies must have identical timestamp/target cohorts.

M1 and full-year M2 cover the same 2017 timestamps but use different fit origins;
the four-week experiment also changes the cohort. Differences between their
aggregate scores are not evidence of a changed model's improvement. Full details
are in the [M1 protocol](evaluation.md), [four-week protocol](temporal-evaluation.md)
and [saved full-year protocol](weather-sensitivity-protocol.json).

### Models and metrics

Every fit uses the same forest: 100 trees, maximum depth 12, minimum leaf size 2,
seed 42 and one worker. A fresh pipeline fits scaling and category discovery on
training records only. Features reuse the existing temperature/cloud, weather,
holiday, hour-category and cyclical calendar representation. No hyperparameter
search or automatic winner selection occurs. There is no weather ablation, so
an advantage over a calendar baseline cannot be assigned specifically to weather.

The global-mean baseline uses only training targets. The weekday/hour baseline
uses training group means, falling back to the training global mean when a group
is absent. M2 additionally looks up the target exactly seven naive calendar days
before the prediction timestamp, from the training set only. Missing exact lags
use the calendar baseline, then the global mean; zero is a valid exact match.
A row shift or nearest timestamp is not a substitute for this lookup on an
irregular series. Validation windows of at most seven days keep every possible
exact lag before the fit origin. The lag baseline also remains retrospective
because label publication latency is unknown.

MAE describes average absolute error in vehicles/hour; RMSE gives large residuals
more weight. Pooled metrics use all individual residuals, not an unweighted mean
of fold metrics. M1 additionally retains R² and monthly results in its raw report.
All tabulated comparisons below are within the same experiment and cohort.
No confidence interval or significance test is claimed.

## Recorded results

All values below come from the original committed JSON reports, rounded to three
decimal places. Their original environment was Python 3.13.14 on macOS arm64.
This is historical attribution, not a recommendation to use that interpreter.
Current reproduction must meet [SECURITY.md](../SECURITY.md)'s patched-runtime
requirement. The recorded evaluation stack is NumPy 2.3.4, pandas 2.3.3,
scikit-learn 1.7.2, SciPy 1.18.1, joblib 1.6.0 and threadpoolctl 3.7.0.
The original reports do not identify CPU model or RAM; this report makes no
runtime-performance comparison.

### M1: one fit, all retained 2017 hours

| Model | MAE | RMSE |
|---|---:|---:|
| Training mean | 1744.392 | 1986.302 |
| Weekday/hour mean | 278.654 | 460.997 |
| Random forest | 258.390 | 436.333 |

The calendar baseline is substantially more informative here than a global mean.
Comparing only with the latter would conceal most of that baseline's predictive
value. The forest has lower aggregate MAE and RMSE than the calendar baseline
on this recorded cohort. This does not establish future or multi-road quality.
[Full M1 report and monthly breakdowns](../benchmarks/validation-2017.json).

### M2: four development weeks

| Model | MAE | RMSE |
|---|---:|---:|
| Training mean | 1642.860 | 1894.804 |
| Weekday/hour mean | 355.604 | 645.721 |
| Seasonal seven-day | 355.814 | 623.410 |
| Random forest | 314.524 | 628.781 |

The forest has lower pooled MAE, while the seasonal baseline has lower pooled
RMSE: 623.410 versus 628.781. Choosing just the more favorable metric would hide
this trade-off. The seasonal baseline has 665 exact matches and one calendar
fallback. [Every fold and the 666 timestamped predictions](../benchmarks/temporal-development-2017.json)
remain available, including [the original fold summary](../benchmarks/temporal-development-2017.md).

### M2: full-year paired weather-row sensitivity

Both training and validation select either the first or the last valid-temperature
weather row in source order. This changes features at 4,293 of 34,032 development
timestamps. Of the 1,349 changed 2017 timestamps, 1,348 change weather category
and one changes temperature. No 2017 cloud or holiday value changes. Training
features also change, so prediction differences cannot be attributed solely to
changed validation features. First and last are two alternatives, not exhaustive
bounds over possible weather representations.

| Model / row policy | MAE | RMSE |
|---|---:|---:|
| Training mean / both | 1743.144 | 1985.385 |
| Weekday/hour mean / both | 272.328 | 456.332 |
| Seasonal seven-day / both | 301.986 | 563.093 |
| Random forest / first | 253.586 | 434.444 |
| Random forest / last | 253.030 | 434.151 |

All baseline predictions are identical between policies. The seasonal baseline
uses 8,667 exact matches, 46 calendar fallbacks and no global fallback. The paired
last-minus-first forest differences are −0.556 MAE and −0.293 RMSE. These small
aggregate differences do not select a policy or establish statistical significance.

| Quarter | Observations | Calendar MAE | Seasonal MAE | First-row forest MAE | Last-row forest MAE |
|---|---:|---:|---:|---:|---:|
| Q1 | 2,141 | 279.848 | 307.361 | 250.423 | 248.531 |
| Q2 | 2,175 | 238.339 | 256.680 | 199.798 | 202.125 |
| Q3 | 2,197 | 262.313 | 287.156 | 251.661 | 252.432 |
| Q4 | 2,200 | 308.616 | 356.357 | 311.765 | 308.333 |

The first-row forest loses to the calendar baseline on Q4 MAE (311.765 versus
308.616), despite winning the pooled comparison. Last-row forest MAE is worse
than first-row MAE in Q2 and Q3. Q1 RMSE also increases slightly, from 399.311 to
399.441. These observations reject a uniform policy advantage. Every quarter's
MAE/RMSE, all 53 folds per policy and every prediction remain in the
[full sensitivity report](../benchmarks/weather-sensitivity-2017.json).

## Implementation checks and their limits

[Temporal tests](../tests/test_temporal_evaluation.py) check fixed boundaries,
unique timestamps, exact zero-valued lags, missing-lag fallbacks, fresh training
pipelines and validation-label mutations. [Sensitivity tests](../tests/test_weather_sensitivity.py)
check complete nonoverlapping coverage, identical paired cohorts/baselines,
reserved-value isolation and metrics recomputed from saved residuals. They use
synthetic fixtures to test behavior, rather than accepting one favorable score.

[Provenance tests](../tests/test_evaluation_failures.py) exercise changes to data,
source, locks, dependency metadata and protocol during M2 evaluation. Detected
boundary drift aborts output; exclusive output creation prevents overwriting an
existing report. M1 has its [own narrower checks](../tests/test_m1_provenance.py).
These mechanisms do not attest loaded code or dependency binaries and cannot
detect changes that are reverted between boundary reads.

M2 reports contain implementation, lock and protocol hashes and raw predictions.
M1 records dataset/configuration/environment and prediction digests but does not
retain individual predictions or implementation/lock hashes. Do not treat its
provenance contract as identical to M2's. Later hardening preserves the historical
reports; current runs record current source hashes rather than rewriting history.

## Reproduction and artifact identity

Use the [maintained source-checkout setup](../README.md#maintained-setup), verify
the actual patched Python interpreter, then run from the repository root with
new output filenames:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output report-m1.json
.venv/bin/python -m src.temporal_evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output report-four-weeks.json
.venv/bin/python -m src.weather_sensitivity --data data/Metro_Interstate_Traffic_Volume.csv --output report-weather-rows.json
```

Installation needs package access; these evaluations need no network service,
Cassandra, API key or saved pickle. Full-year sensitivity fits 106 forests with
one worker each. Keep generated reports separate from the committed artifacts.
The [unchanged evaluation workflow](../.github/workflows/evaluation.yml) reproduces
all three protocols and checks cohort identity, cleaning, baseline accounting
and current provenance. Cross-platform floating-point byte identity is not a
requirement. Runtime checks on a newer Python version do not change the original
reports' recorded environment or make a new experiment.

| Frozen report | SHA-256 |
|---|---|
| [M1](../benchmarks/validation-2017.json) | `35d999d82b53ac3b260e0d37f18e85b27378951bd8101d5de4b971e6545731ef` |
| [Four weeks](../benchmarks/temporal-development-2017.json) | `7ab38803a669611d4f10ec94edb395b105fa0f2618c64f1e994cd9f8c1033fd1` |
| [Weather rows](../benchmarks/weather-sensitivity-2017.json) | `8c135e59f49affc696e3cb2d1782a248929f8b4afc27f7cf5402328c4b0ee8c4` |

## What follows from this evidence

Within the recorded development experiments, calendar structure provides a
strong baseline, metric choice changes one model comparison, and aggregate
results hide a quarter where the calendar baseline has lower MAE. Weather-row
choice has a small pooled effect here, but its direction varies over time. This
supports retaining meaningful baselines, timestamp identity, coverage accounting
and paired cohorts when maintaining the example. It does not justify choosing a
new row policy from these results.

The evidence remains limited to one historical station, one reused development
year, fixed model settings and two source-order weather choices. Missing hours,
naive clocks, unknown input/label latency and unquantified temporal dependence
limit interpretation. There is no confidence interval, weather-feature ablation,
independent holdout result, deployment validation or usage/adoption claim.
**2018 stays unscored; this report does not authorize opening that partition.**
