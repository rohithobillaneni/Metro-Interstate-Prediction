"""Paired weather-row sensitivity over every retained 2017 development hour."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

from src.evaluation import MODEL_PARAMETERS
from src.offline_data import TrafficRecord, load_development_csv
from src.temporal_evaluation import (
    BASELINES, ROOT, _coverage, _digest, _paired, _runtime_metadata, _scores, _source_hashes,
    evaluate_folds, folds_for_windows,
)


START = datetime(2017, 1, 1)
END = datetime(2018, 1, 1)
POLICIES = ("first", "last")
WINDOWS = tuple(
    (start.isoformat(), min(start + timedelta(days=7), END).isoformat())
    for start in (START + timedelta(days=offset) for offset in range(0, 365, 7))
)
PROTOCOL_PATH = ROOT / "docs" / "weather-sensitivity-protocol.json"


def protocol() -> dict:
    """Describe the fixed design, not a configurable parameter search."""
    return {
        "id": "metro-weather-row-2017-v1",
        "task": "retrospective same-hour traffic estimation using observed weather; not advance forecasting",
        "windows": [{"start": start, "end_exclusive": end} for start, end in WINDOWS],
        "training": "all retained timestamps strictly before each origin; fresh pipeline per fold and policy",
        "weather_row_policies": list(POLICIES),
        "weather_row_semantics": "first or last valid-temperature source row per timestamp; require target agreement",
        "reserved_start": END.isoformat(),
        "reserved_targets_validated": False,
        "reserved_scored": False,
        "timestamp_semantics": "naive whole-hour source clock; no timezone/DST inference or gap filling",
        "seasonal_lag": "exact same-clock-time seven calendar days earlier from training only",
        "seasonal_fallback": "training weekday/hour mean, then training global mean",
        "model_parameters": MODEL_PARAMETERS.copy(),
        "metrics": ["mae", "rmse"],
        "pooling": "every retained 2017 timestamp once; pool residuals, not fold metrics",
        "temporal_cohorts": "fixed calendar quarters of 2017, descriptive only",
        "paired_mae_sign": "random_forest minus baseline on the same observations",
        "sensitivity_sign": "last minus first on the same observations; no policy selection",
        "limitations": "development year, not untouched test data; first/last are alternatives, not exhaustive bounds; operational feature/label latency unknown; no significance claim",
    }


def full_year_folds(records: tuple[TrafficRecord, ...]):
    folds = folds_for_windows(records, WINDOWS)
    expected = [r.date_time for r in sorted(records, key=lambda r: r.date_time)
                if START <= r.date_time < END]
    actual = [r.date_time for fold in folds for r in fold.validation]
    if actual != expected:
        raise ValueError("Full-year folds must cover every eligible timestamp exactly once")
    return folds


def feature_changes(first: tuple[TrafficRecord, ...], last: tuple[TrafficRecord, ...]) -> dict:
    if [(r.date_time, r.traffic_volume) for r in first] != [(r.date_time, r.traffic_volume) for r in last]:
        raise ValueError("Weather policies must retain identical ordered timestamps and targets")
    fields = ("temp", "clouds_all", "weather_main", "holiday")
    result = {}
    for name, pairs in (
        ("development", list(zip(first, last))),
        ("validation_2017", [(a, b) for a, b in zip(first, last) if START <= a.date_time < END]),
    ):
        result[name] = {
            "count": len(pairs),
            "changed_rows": sum(any(getattr(a, f) != getattr(b, f) for f in fields) for a, b in pairs),
            "changed_fields": {f: sum(getattr(a, f) != getattr(b, f) for a, b in pairs) for f in fields},
        }
    return result


def summarize(rows: list[dict]) -> dict:
    actual = [r["actual"] for r in rows]
    scores = _scores(actual, {name: [r["predictions"][name] for r in rows]
                              for name in (*BASELINES, "random_forest")})
    return {"count": len(rows), "metrics": scores, "paired_mae_differences": _paired(scores)}


def paired_differences(first: list[dict], last: list[dict]) -> dict:
    if [(r["date_time"], r["actual"]) for r in first] != [(r["date_time"], r["actual"]) for r in last]:
        raise ValueError("Policy reports must use identical timestamps and targets")
    for a, b in zip(first, last):
        if any(a["predictions"][name] != b["predictions"][name] for name in BASELINES):
            raise ValueError("Weather-independent baselines changed between policies")
    first_scores, last_scores = summarize(first)["metrics"], summarize(last)["metrics"]
    return {name: {metric: last_scores[name][metric] - first_scores[name][metric]
                   for metric in ("mae", "rmse")}
            for name in first_scores}


def source_hashes() -> dict:
    return {**_source_hashes(), "src/weather_sensitivity.py": _digest(Path(__file__).read_bytes())}


def evaluate(path: str | Path) -> dict:
    path = Path(path)
    source_hash, code_hashes = _digest(path.read_bytes()), source_hashes()
    metadata = _runtime_metadata()
    declared = protocol()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    if json.loads(protocol_bytes) != declared:
        raise ValueError("Implementation differs from the saved experiment protocol")
    loaded = {policy: load_development_csv(path, weather_row=policy) for policy in POLICIES}
    if loaded["first"][1] != loaded["last"][1]:
        raise ValueError("Weather policies changed cleaning counts")
    changes = feature_changes(loaded["first"][0], loaded["last"][0])
    if _digest(path.read_bytes()) != source_hash:
        raise ValueError("Input CSV changed while loading")
    policy_reports = {}
    predictions = {}
    for policy in POLICIES:
        records = loaded[policy][0]
        fold_reports, _, _ = evaluate_folds(full_year_folds(records))
        rows = [row for fold in fold_reports for row in fold["predictions"]]
        predictions[policy] = rows
        quarter_reports = {}
        for quarter in range(1, 5):
            quarter_rows = [r for r in rows if (datetime.fromisoformat(r["date_time"]).month - 1) // 3 + 1 == quarter]
            quarter_reports[f"Q{quarter}"] = summarize(quarter_rows)
        policy_reports[policy] = {
            "folds": fold_reports, "pooled": summarize(rows), "quarters": quarter_reports,
            "coverage": _coverage(tuple(r for r in records if START <= r.date_time < END), START, END),
        }
    difference = paired_differences(predictions["first"], predictions["last"])
    if (_digest(path.read_bytes()) != source_hash or source_hashes() != code_hashes
            or PROTOCOL_PATH.read_bytes() != protocol_bytes):
        raise ValueError("Input CSV, protocol or implementation changed during evaluation")
    if _runtime_metadata() != metadata or protocol() != declared:
        raise ValueError("Runtime metadata, dependency lock or protocol changed during evaluation")
    return {
        "schema_version": 1,
        "protocol": declared,
        "protocol_sha256": _digest(json.dumps(declared, sort_keys=True, separators=(",", ":")).encode()),
        "protocol_file_sha256": _digest(protocol_bytes),
        "source_sha256": source_hash,
        "source_code_sha256": code_hashes,
        "evaluation_lock_sha256": metadata["evaluation_lock_sha256"],
        "cleaning": asdict(loaded["first"][1]),
        "feature_changes": changes,
        "policies": policy_reports,
        "last_minus_first_metrics": difference,
        "environment": metadata["environment"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
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
