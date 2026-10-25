"""Offline chronological evaluation; never imports the legacy training scripts."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.data_transformation import get_preprocessing_pipeline
from src.offline_data import TrafficRecord, load_csv, partition_records


MODEL_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": 1,
}


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    """Compute same-cohort metrics; undefined constant-target R2 is JSON null."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metrics require equal, nonempty target and prediction lists")
    if any(not math.isfinite(value) for value in actual + predicted):
        raise ValueError("Targets and predictions must be finite")
    errors = [target - estimate for target, estimate in zip(actual, predicted)]
    mse = math.fsum(error * error for error in errors) / len(errors)
    mean = math.fsum(actual) / len(actual)
    variance = math.fsum((value - mean) ** 2 for value in actual) / len(actual)
    return {
        "mae": math.fsum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(mse),
        "r2": None if variance == 0 else 1 - mse / variance,
    }


def baseline_predictions(
    train: tuple[TrafficRecord, ...], validation: tuple[TrafficRecord, ...]
) -> tuple[dict[str, list[float]], int]:
    """Fit a global mean and weekday/hour means using training targets only."""
    if not train:
        raise ValueError("Baselines require nonempty training data")
    mean = math.fsum(record.traffic_volume for record in train) / len(train)
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in train:
        groups[record.date_time.weekday(), record.date_time.hour].append(record.traffic_volume)
    calendar = {key: math.fsum(values) / len(values) for key, values in groups.items()}
    keys = [(record.date_time.weekday(), record.date_time.hour) for record in validation]
    return {
        "training_mean": [mean] * len(validation),
        "weekday_hour_mean": [calendar.get(key, mean) for key in keys],
    }, sum(key not in calendar for key in keys)


def make_model() -> Pipeline:
    """Reuse the original feature transformer, fitted inside the training pipeline."""
    return Pipeline([
        ("preprocessor", get_preprocessing_pipeline()),
        ("model", RandomForestRegressor(**MODEL_PARAMETERS)),
    ])


def _cohort(records: tuple[TrafficRecord, ...]) -> dict:
    return {
        "count": len(records),
        "first_timestamp": records[0].date_time.isoformat() if records else None,
        "last_timestamp": records[-1].date_time.isoformat() if records else None,
    }


def _environment() -> dict:
    """Read runtime and installed-version metadata, not loaded-code identity."""
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "dependencies": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
        },
    }


def evaluate(path: str | Path) -> dict:
    """Evaluate calendar 2017; reserve 2018 onward without fitting or scoring it."""
    path = Path(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    environment = _environment()
    model_parameters = MODEL_PARAMETERS.copy()
    records, diagnostics = load_csv(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Input CSV changed while loading")
    partitions = partition_records(records)
    train, validation = partitions.train, partitions.validation
    predictions, fallbacks = baseline_predictions(train, validation)
    model = make_model()
    model.fit(
        pd.DataFrame([record.features() for record in train]),
        [record.traffic_volume for record in train],
    )
    predictions["random_forest"] = model.predict(
        pd.DataFrame([record.features() for record in validation])
    ).tolist()
    actual = [record.traffic_volume for record in validation]
    results = {}
    for name, predicted in predictions.items():
        results[name] = {
            **metrics(actual, predicted),
            "prediction_sha256": hashlib.sha256(
                json.dumps(predicted, allow_nan=False, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    monthly = {}
    for month in sorted({record.date_time.month for record in validation}):
        indices = [i for i, record in enumerate(validation) if record.date_time.month == month]
        monthly[str(month)] = {
            "count": len(indices),
            "metrics": {
                name: metrics([actual[i] for i in indices], [values[i] for i in indices])
                for name, values in predictions.items()
            },
        }
    # Keep the existing report bound to its initial snapshot. These checks
    # detect persistent boundary drift, not mutations reverted between reads.
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Input CSV changed during evaluation")
    if _environment() != environment:
        raise ValueError("Runtime metadata changed during evaluation")
    if MODEL_PARAMETERS != model_parameters:
        raise ValueError("Model parameters changed during evaluation")
    return {
        "schema_version": 1,
        "task": "same-hour traffic estimation with observed weather; not advance forecasting",
        "source_sha256": source_hash,
        "cleaning": asdict(diagnostics),
        "split": {
            "validation_start": "2017-01-01",
            "reserved_test_start": "2018-01-01",
            "train": _cohort(train),
            "validation": _cohort(validation),
            "reserved_test": {**_cohort(partitions.test), "scored": False, "used_for_fit": False},
        },
        "weekday_hour_fallback_count": fallbacks,
        "model_parameters": model_parameters,
        "metrics": results,
        "validation_months": monthly,
        "environment": environment,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="UCI-format raw CSV")
    parser.add_argument("--output", type=Path, help="New JSON result file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new path to preserve previous results")
    try:
        report = evaluate(args.data)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
