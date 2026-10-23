"""Fixed development-only temporal validation; reserved 2018 is never scored."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from typing import Iterable

import pandas as pd

from src.evaluation import MODEL_PARAMETERS, baseline_predictions, make_model, metrics
from src.offline_data import TrafficRecord, load_development_csv, partition_records


DEVELOPMENT_WINDOWS = (
    ("2017-01-01", "2017-01-08"),
    ("2017-04-01", "2017-04-08"),
    ("2017-07-01", "2017-07-08"),
    ("2017-10-01", "2017-10-08"),
)
BASELINES = ("training_mean", "weekday_hour_mean", "seasonal_7_day")
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Fold:
    start: datetime
    end: datetime
    train: tuple[TrafficRecord, ...]
    validation: tuple[TrafficRecord, ...]


def _check_timestamps(records: tuple[TrafficRecord, ...]) -> None:
    timestamps = [record.date_time for record in records]
    if any(stamp.tzinfo is not None for stamp in timestamps):
        raise ValueError("Records must have naive source timestamps")
    if any(stamp.minute or stamp.second or stamp.microsecond for stamp in timestamps):
        raise ValueError("Temporal evaluation requires whole-hour timestamps")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Records must have unique timestamps")
    if any(stamp >= datetime(2018, 1, 1) for stamp in timestamps):
        raise ValueError("Temporal evaluation accepts only pre-2018 development records")


def temporal_folds(records: Iterable[TrafficRecord]) -> tuple[Fold, ...]:
    """Use four fixed seven-day windows with strictly earlier expanding training."""
    supplied = tuple(records)
    _check_timestamps(supplied)
    folds = []
    for start, end in DEVELOPMENT_WINDOWS:
        partitions = partition_records(supplied, start, end)
        folds.append(Fold(
            datetime.fromisoformat(start), datetime.fromisoformat(end),
            partitions.train, partitions.validation,
        ))
    return tuple(folds)


def seasonal_predictions(
    train: tuple[TrafficRecord, ...], validation: tuple[TrafficRecord, ...]
) -> tuple[list[float], dict[str, int]]:
    """Look up exactly seven naive calendar days earlier, never a nearby row.

    Missing lags use the training weekday/hour mean, then global training mean.
    This is a retrospective comparison: operational label latency is unknown.
    Validation labels are never read. Predictions retain the supplied order.
    """
    if not train or not validation:
        raise ValueError("Train and validation must both be nonempty")
    _check_timestamps(train)
    _check_timestamps(validation)
    if max(row.date_time for row in train) >= min(row.date_time for row in validation):
        raise ValueError("All training timestamps must precede validation")
    calendar, _ = baseline_predictions(train, validation)
    lookup = {row.date_time: row.traffic_volume for row in train}
    calendar_keys = {(row.date_time.weekday(), row.date_time.hour) for row in train}
    counts = {"exact_lag": 0, "calendar_fallback": 0, "global_fallback": 0}
    predictions = []
    for index, row in enumerate(validation):
        lag = row.date_time - timedelta(days=7)
        if lag in lookup:
            predictions.append(lookup[lag])
            counts["exact_lag"] += 1
        else:
            predictions.append(calendar["weekday_hour_mean"][index])
            key = row.date_time.weekday(), row.date_time.hour
            counts["calendar_fallback" if key in calendar_keys else "global_fallback"] += 1
    return predictions, counts


def _coverage(records: tuple[TrafficRecord, ...], start: datetime, end: datetime) -> dict:
    """Count absent naive clock slots, including both interval edges, without filling them."""
    expected = (end - start) // timedelta(hours=1)
    positions = sorted((row.date_time - start) // timedelta(hours=1) for row in records)
    previous, longest = -1, 0
    for position in positions:
        longest = max(longest, position - previous - 1)
        previous = position
    longest = max(longest, expected - previous - 1)
    return {
        "observed_hours": len(records),
        "expected_clock_hours": expected,
        "missing_clock_hours": expected - len(records),
        "longest_missing_run_hours": longest,
    }


def _cohort(records: tuple[TrafficRecord, ...]) -> dict:
    return {
        "count": len(records),
        "first_timestamp": records[0].date_time.isoformat(),
        "last_timestamp": records[-1].date_time.isoformat(),
    }


def _scores(actual: list[float], predictions: dict[str, list[float]]) -> dict:
    return {
        name: {key: value for key, value in metrics(actual, values).items() if key in ("mae", "rmse")}
        for name, values in predictions.items()
    }


def _paired(scores: dict) -> dict[str, float]:
    return {
        baseline: scores["random_forest"]["mae"] - scores[baseline]["mae"]
        for baseline in BASELINES
    }


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "src/" + name: _digest((ROOT / "src" / name).read_bytes())
        for name in ("temporal_evaluation.py", "offline_data.py", "evaluation.py", "data_transformation.py")
    }


def evaluate(path: str | Path) -> dict:
    """Fit each fixed fold independently and retain every prediction and comparison."""
    path = Path(path)
    source_hash = _digest(path.read_bytes())
    code_hashes = _source_hashes()
    records, diagnostics = load_development_csv(path)
    if _digest(path.read_bytes()) != source_hash:
        raise ValueError("Input CSV changed while loading")
    folds = temporal_folds(records)
    protocol = {
        "id": "metro-temporal-development-v1",
        "task": "retrospective same-hour traffic estimation with observed weather; not advance forecasting",
        "windows": [{"start": start, "end_exclusive": end} for start, end in DEVELOPMENT_WINDOWS],
        "training": "all retained timestamps strictly before each validation start",
        "reserved_start": "2018-01-01",
        "reserved_targets_validated": False,
        "reserved_scored": False,
        "timestamp_semantics": "naive local clock; whole-hour grid; no timezone or DST inference",
        "seasonal_lag": "exact same-clock-time seven calendar days earlier from training only",
        "seasonal_fallback": "training weekday/hour mean, then training global mean",
        "paired_mae_sign": "random_forest minus baseline; negative favors forest on the same cohort",
        "model_parameters": MODEL_PARAMETERS.copy(),
        "limitations": "four development weeks, not annual performance; no significance or operational availability claim",
    }
    fold_reports = []
    pooled_actual: list[float] = []
    pooled_predictions: dict[str, list[float]] = {name: [] for name in (*BASELINES, "random_forest")}
    for fold in folds:
        predictions, calendar_fallbacks = baseline_predictions(fold.train, fold.validation)
        predictions["seasonal_7_day"], seasonal_counts = seasonal_predictions(fold.train, fold.validation)
        model = make_model()
        model.fit(
            pd.DataFrame([row.features() for row in fold.train]),
            [row.traffic_volume for row in fold.train],
        )
        predictions["random_forest"] = model.predict(
            pd.DataFrame([row.features() for row in fold.validation])
        ).tolist()
        actual = [row.traffic_volume for row in fold.validation]
        scores = _scores(actual, predictions)
        fold_reports.append({
            "validation_start": fold.start.isoformat(),
            "validation_end": fold.end.isoformat(),
            "train": _cohort(fold.train),
            "validation": _cohort(fold.validation),
            "coverage": {
                "train": _coverage(fold.train, fold.train[0].date_time, fold.start),
                "validation": _coverage(fold.validation, fold.start, fold.end),
            },
            "baselines": {
                "weekday_hour_mean": {"global_fallback": calendar_fallbacks},
                "seasonal_7_day": seasonal_counts,
            },
            "metrics": scores,
            "paired_mae_differences": _paired(scores),
            "predictions": [
                {
                    "date_time": row.date_time.isoformat(),
                    "actual": actual[index],
                    "predictions": {name: values[index] for name, values in predictions.items()},
                }
                for index, row in enumerate(fold.validation)
            ],
        })
        pooled_actual.extend(actual)
        for name, values in predictions.items():
            pooled_predictions[name].extend(values)
    if _digest(path.read_bytes()) != source_hash or _source_hashes() != code_hashes:
        raise ValueError("Input CSV or implementation changed during evaluation")
    pooled = _scores(pooled_actual, pooled_predictions)
    return {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": _digest(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()),
        "source_sha256": source_hash,
        "source_code_sha256": code_hashes,
        "evaluation_lock_sha256": _digest((ROOT / "requirements-evaluation.txt").read_bytes()),
        "cleaning": asdict(diagnostics),
        "folds": fold_reports,
        "pooled_count": len(pooled_actual),
        "pooled_metrics": pooled,
        "paired_mae_differences": _paired(pooled),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "dependencies": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="UCI-format raw CSV")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new path to preserve previous results")
    try:
        encoded = json.dumps(evaluate(args.data), indent=2, allow_nan=False) + "\n"
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
