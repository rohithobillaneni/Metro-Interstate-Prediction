# Fixed 2017 validation comparison

`validation-2017.json` is the complete first run of the
[prespecified offline protocol](../docs/evaluation.md), executed on 2026-10-23
with Python 3.13.14 on macOS ARM64. The exact installed core dependency versions,
model parameters, source hash, counts, monthly metrics and prediction digests
are in the JSON. The environment was an isolated virtual environment using
`requirements-evaluation.txt`. No hyperparameter search or reranking of models
was performed after seeing these results.

Command from the repository root:

```sh
.venv/bin/python -m src.evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output validation-result.json
```

Training has 25,319 timestamps before 2017; validation has 8,713 timestamps during 2017.
The 6,533 timestamps from 2018 remain unscored and unused for fitting.

| Model | MAE (vehicles/hour) | RMSE (vehicles/hour) | R2 |
|---|---:|---:|---:|
| Training mean | 1744.39 | 1986.30 | -0.00387 |
| Training weekday/hour mean | 278.65 | 461.00 | 0.94593 |
| Seeded random forest | 258.39 | 436.33 | 0.95156 |

The forest has lower aggregate error on this validation cohort. It has **higher
monthly MAE in July, November and December** than the calendar baseline. This
limited comparison establishes neither general superiority nor deployment
readiness. It does not compare against an earlier stored score from the
repository, whose cohort and validation protocol differ. Runtime and memory
performance were not benchmarked. The calendar model required no fallback for
unseen weekday/hour combinations in this dataset.

The result uses one fixed split and single seeded fit, with no uncertainty
interval. Future rolling-origin evaluation and availability-aware seasonal
baselines remain necessary before stronger forecasting or generalization
claims. Preserve these results, including the unfavorable months.
