import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src import temporal_evaluation
from src.evaluation import make_model
from src.offline_data import TrafficRecord, load_development_csv


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (
    ("2017-01-01", "2017-01-08"),
    ("2017-04-01", "2017-04-08"),
    ("2017-07-01", "2017-07-08"),
    ("2017-10-01", "2017-10-08"),
)
MODEL_NAMES = {"training_mean", "weekday_hour_mean", "seasonal_7_day", "random_forest"}


def record(timestamp, target=10):
    return TrafficRecord(datetime.fromisoformat(timestamp), target, 15, 40, "Clouds", 0)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TemporalFoldTests(unittest.TestCase):
    def setUp(self):
        self.records = [record("2016-12-31T23:00:00")]
        for start, end in WINDOWS:
            self.records.extend((
                record(start),
                replace(record(end), date_time=datetime.fromisoformat(end) - timedelta(hours=1)),
                record(end),
            ))

    def test_fixed_calendar_boundaries_and_expanding_training(self):
        self.assertEqual(temporal_evaluation.DEVELOPMENT_WINDOWS, WINDOWS)
        folds = temporal_evaluation.temporal_folds(reversed(self.records))
        self.assertEqual(len(folds), 4)
        for index, (fold, (start, end)) in enumerate(zip(folds, WINDOWS)):
            with self.subTest(start=start):
                self.assertEqual(fold.start, datetime.fromisoformat(start))
                self.assertEqual(fold.end, datetime.fromisoformat(end))
                self.assertEqual([r.date_time for r in fold.validation], [
                    fold.start, fold.end - timedelta(hours=1),
                ])
                expected_train = sorted(
                    (r for r in self.records if r.date_time < fold.start),
                    key=lambda r: r.date_time,
                )
                self.assertEqual(fold.train, tuple(expected_train))
                self.assertTrue(all(r.date_time < fold.start for r in fold.train))
                if index:
                    self.assertTrue(set(folds[index - 1].validation).issubset(fold.train))
                self.assertNotIn(record(end), fold.validation)

    def test_partition_membership_does_not_inspect_target_values(self):
        changed = [replace(r, traffic_volume="not a numeric label") for r in self.records]
        original_folds = temporal_evaluation.temporal_folds(self.records)
        changed_folds = temporal_evaluation.temporal_folds(changed)
        for original, altered in zip(original_folds, changed_folds):
            self.assertEqual([r.date_time for r in original.train], [r.date_time for r in altered.train])
            self.assertEqual([r.date_time for r in original.validation], [r.date_time for r in altered.validation])

    def test_each_empty_validation_window_fails_without_moving_it(self):
        for start, end in WINDOWS:
            with self.subTest(start=start):
                remaining = [r for r in self.records if not (
                    datetime.fromisoformat(start) <= r.date_time < datetime.fromisoformat(end)
                )]
                with self.assertRaises(ValueError):
                    temporal_evaluation.temporal_folds(remaining)

    def test_empty_training_and_empty_input_fail(self):
        for records in ([], self.records[1:]):
            with self.subTest(count=len(records)):
                with self.assertRaises(ValueError):
                    temporal_evaluation.temporal_folds(records)

    def test_duplicate_aware_non_hourly_and_reserved_timestamps_fail(self):
        for invalid in (
            self.records[0],
            replace(self.records[0], date_time=self.records[0].date_time.replace(tzinfo=timezone.utc)),
            record("2016-12-31T23:30:00"),
            record("2016-12-31T23:00:01"),
            record("2018-01-01T00:00:00"),
        ):
            with self.subTest(timestamp=invalid.date_time):
                with self.assertRaises(ValueError):
                    temporal_evaluation.temporal_folds([*self.records, invalid])


class SeasonalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.train = (
            record("2016-12-18T08:00:00", 20),
            record("2016-12-25T08:00:00", 0),
            record("2016-12-19T09:00:00", 40),
            record("2016-12-12T09:00:00", 60),
            record("2016-12-27T09:00:00", 80),
            record("2016-12-26T08:00:00", 160),
        )
        self.validation = (
            record("2017-01-01T08:00:00", 1000),
            record("2017-01-02T09:00:00", 2000),
            record("2017-01-03T10:00:00", 3000),
        )

    def test_exact_zero_and_missing_hour_fallbacks_use_training_only(self):
        predictions, counts = temporal_evaluation.seasonal_predictions(self.train, self.validation)
        # Monday's exact prior-week hour is absent; the nearby 08:00 row is not a substitute.
        self.assertEqual(predictions, [0, 50, 60])
        self.assertEqual(counts, {"exact_lag": 1, "calendar_fallback": 1, "global_fallback": 1})

    def test_validation_labels_do_not_affect_predictions_or_coverage(self):
        original = temporal_evaluation.seasonal_predictions(self.train, self.validation)
        changed = tuple(replace(r, traffic_volume=999999) for r in self.validation)
        self.assertEqual(temporal_evaluation.seasonal_predictions(self.train, changed), original)

    def test_prediction_order_matches_supplied_validation_order(self):
        predictions, counts = temporal_evaluation.seasonal_predictions(self.train, tuple(reversed(self.validation)))
        self.assertEqual(predictions, [60, 50, 0])
        self.assertEqual(sum(counts.values()), 3)

    def test_naive_local_clock_lookup_does_not_infer_dst_conversion(self):
        predictions, counts = temporal_evaluation.seasonal_predictions(
            (record("2017-03-12T02:00:00", 22),),
            (record("2017-03-19T02:00:00", 33),),
        )
        self.assertEqual(predictions, [22])
        self.assertEqual(counts["exact_lag"], 1)

    def test_empty_training_fails(self):
        with self.assertRaises(ValueError):
            temporal_evaluation.seasonal_predictions((), self.validation)

    def test_empty_validation_or_non_past_training_fails(self):
        with self.assertRaises(ValueError):
            temporal_evaluation.seasonal_predictions(self.train, ())
        for timestamp in ("2017-01-01T08:00:00", "2017-01-01T09:00:00"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    temporal_evaluation.seasonal_predictions(
                        (*self.train, record(timestamp)), self.validation,
                    )


class TemporalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture_directory.cleanup)
        cls.fixture_path = Path(cls.fixture_directory.name) / "source.csv"
        cls.rows = []
        for index, timestamp in enumerate((
            "2016-12-11 08:00:00", "2016-12-18 08:00:00",
            "2016-12-12 09:00:00", "2016-12-19 09:00:00",
            "2016-12-13 10:00:00", "2016-12-20 10:00:00",
        )):
            cls.rows.append(cls.row(timestamp, 10 + index * 20, 273.15 + index, 10 + index, "Clouds"))
        for index, (start, end) in enumerate(WINDOWS):
            origin = datetime.fromisoformat(start)
            cls.rows.append(cls.row(
                (origin - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
                index * 100, 280 + index, 20 + index, f"History-{index}",
            ))
            for hour in range(index + 2):
                cls.rows.append(cls.row(
                    (origin + timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S"),
                    1000 + index * 1000 + hour * 50, 320 + index, 80 + index,
                    f"Validation-{index}",
                ))
            cls.rows.append(cls.row(end + " 00:00:00", 700 + index, 290, 50, "After-window"))
        cls.rows.append({**cls.rows[0], "weather_main": "Mist"})
        cls.rows.append(cls.row("2018-01-01 00:00:00", 9999, 300, 50, "Snow"))
        write_csv(cls.fixture_path, cls.rows)
        cls.report = temporal_evaluation.evaluate(cls.fixture_path)

    @staticmethod
    def row(timestamp, target, temp, clouds, weather):
        return {
            "date_time": timestamp, "traffic_volume": target,
            "temp": temp, "clouds_all": clouds,
            "weather_main": weather, "holiday": "None",
        }

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "source.csv"
        write_csv(self.path, self.rows)

    def test_report_contains_same_cohort_predictions_for_all_models(self):
        self.assertEqual(self.report["schema_version"], 1)
        self.assertIsInstance(self.report["protocol"], dict)
        self.assertEqual(len(self.report["folds"]), 4)
        self.assertEqual(self.report["source_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(self.report["cleaning"], {
            "raw_count": 30, "invalid_temp_dropped": 0, "duplicate_extras": 1,
            "distinct_timestamps": 28, "reserved_rows_excluded": 1,
        })
        for index, fold in enumerate(self.report["folds"]):
            with self.subTest(fold=index):
                self.assertEqual(datetime.fromisoformat(fold["validation_start"]), datetime.fromisoformat(WINDOWS[index][0]))
                self.assertEqual(datetime.fromisoformat(fold["validation_end"]), datetime.fromisoformat(WINDOWS[index][1]))
                self.assertEqual(fold["train"]["count"], (7, 11, 16, 22)[index])
                self.assertEqual(fold["validation"]["count"], index + 2)
                self.assertEqual(set(fold["metrics"]), MODEL_NAMES)
                self.assertEqual(len(fold["predictions"]), index + 2)
                timestamps = [item["date_time"] for item in fold["predictions"]]
                self.assertEqual(timestamps, sorted(timestamps))
                self.assertEqual(len(set(timestamps)), len(timestamps))
                for item in fold["predictions"]:
                    self.assertEqual(set(item["predictions"]), MODEL_NAMES)
                    self.assertTrue(all(math.isfinite(value) for value in item["predictions"].values()))
                    self.assertLess(datetime.fromisoformat(item["date_time"]), datetime(2018, 1, 1))
                    source_row = next(row for row in self.rows
                                      if datetime.fromisoformat(row["date_time"]) == datetime.fromisoformat(item["date_time"]))
                    self.assertEqual(item["actual"], source_row["traffic_volume"])
                self.assertEqual(sum(fold["baselines"]["seasonal_7_day"].values()), index + 2)
                self.assertEqual(fold["baselines"]["seasonal_7_day"]["exact_lag"], 1)
                self.assertEqual(set(fold["baselines"]["weekday_hour_mean"]), {"global_fallback"})
        json.dumps(self.report, allow_nan=False)

    def test_coverage_counts_missing_hours_at_edges_and_between_observations(self):
        for index, fold in enumerate(self.report["folds"]):
            with self.subTest(fold=index):
                self.assertEqual(fold["coverage"]["validation"], {
                    "observed_hours": index + 2,
                    "expected_clock_hours": 168,
                    "missing_clock_hours": 168 - (index + 2),
                    "longest_missing_run_hours": 168 - (index + 2),
                })
                train = fold["coverage"]["train"]
                self.assertEqual(train["observed_hours"], fold["train"]["count"])
                self.assertEqual(train["observed_hours"] + train["missing_clock_hours"], train["expected_clock_hours"])
                self.assertGreater(train["longest_missing_run_hours"], 0)

    def test_protocol_source_lock_and_environment_evidence_matches_executed_inputs(self):
        protocol = self.report["protocol"]
        self.assertEqual(protocol["id"], "metro-temporal-development-v1")
        self.assertEqual(protocol["windows"], [
            {"start": start, "end_exclusive": end} for start, end in WINDOWS
        ])
        self.assertEqual(protocol["reserved_start"], "2018-01-01")
        self.assertIs(protocol["reserved_targets_validated"], False)
        self.assertIs(protocol["reserved_scored"], False)
        self.assertEqual(protocol["model_parameters"], {
            "n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2,
            "random_state": 42, "n_jobs": 1,
        })
        canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(self.report["protocol_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(self.report["source_code_sha256"], {
            "src/" + name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in ("temporal_evaluation.py", "offline_data.py", "evaluation.py", "data_transformation.py")
        })
        self.assertEqual(self.report["evaluation_lock_sha256"],
                         hashlib.sha256((ROOT / "requirements-evaluation.txt").read_bytes()).hexdigest())
        self.assertEqual(self.report["environment"], {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "dependencies": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
            },
        })
        self.assertEqual(self.report["pooled_count"], 14)

    def test_coverage_uses_fixed_window_edges_and_longest_internal_gap(self):
        rows = [dict(row) for row in self.rows if not (
            "2017-01-01" <= row["date_time"] < "2017-01-08"
        )]
        rows.extend(self.row(timestamp, 100, 290, 50, "Clouds") for timestamp in (
            "2017-01-01 10:00:00", "2017-01-01 11:00:00", "2017-01-06 00:00:00",
        ))
        write_csv(self.path, rows)
        report = temporal_evaluation.evaluate(self.path)
        self.assertEqual(report["folds"][0]["coverage"]["validation"], {
            "observed_hours": 3,
            "expected_clock_hours": 168,
            "missing_clock_hours": 165,
            "longest_missing_run_hours": 108,
        })

    def test_pooled_metrics_and_paired_differences_follow_saved_residuals(self):
        all_rows = [row for fold in self.report["folds"] for row in fold["predictions"]]
        self.assertEqual(len(all_rows), 14)
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                residuals = [row["actual"] - row["predictions"][model] for row in all_rows]
                expected_mae = math.fsum(abs(value) for value in residuals) / 14
                expected_rmse = math.sqrt(math.fsum(value * value for value in residuals) / 14)
                self.assertAlmostEqual(self.report["pooled_metrics"][model]["mae"], expected_mae)
                self.assertAlmostEqual(self.report["pooled_metrics"][model]["rmse"], expected_rmse)
                for fold in self.report["folds"]:
                    rows = fold["predictions"]
                    mae = math.fsum(abs(row["actual"] - row["predictions"][model]) for row in rows) / len(rows)
                    self.assertAlmostEqual(fold["metrics"][model]["mae"], mae)
                if model != "random_forest":
                    expected_difference = self.report["pooled_metrics"]["random_forest"]["mae"] - expected_mae
                    self.assertAlmostEqual(self.report["paired_mae_differences"][model], expected_difference)
                    for fold in self.report["folds"]:
                        self.assertAlmostEqual(fold["paired_mae_differences"][model],
                            fold["metrics"]["random_forest"]["mae"] - fold["metrics"][model]["mae"])
        unweighted_fold_mae = sum(fold["metrics"]["training_mean"]["mae"] for fold in self.report["folds"]) / 4
        self.assertNotAlmostEqual(self.report["pooled_metrics"]["training_mean"]["mae"], unweighted_fold_mae)

    def test_each_fold_gets_a_fresh_model_fitted_only_on_its_training_rows(self):
        models = []
        def new_model():
            model = make_model()
            models.append(model)
            return model
        with patch("src.temporal_evaluation.make_model", side_effect=new_model) as factory:
            temporal_evaluation.evaluate(self.path)
        self.assertEqual(factory.call_count, 4)
        self.assertEqual(len({id(model) for model in models}), 4)
        records, _ = load_development_csv(self.path)
        for model, fold in zip(models, temporal_evaluation.temporal_folds(records)):
            transformers = model.named_steps["preprocessor"].named_transformers_
            scaler = transformers["num"].named_steps["scaler"]
            self.assertEqual(scaler.n_samples_seen_, len(fold.train))
            self.assertAlmostEqual(scaler.mean_[0], math.fsum(r.temp for r in fold.train) / len(fold.train))
            self.assertAlmostEqual(scaler.mean_[1], math.fsum(r.clouds_all for r in fold.train) / len(fold.train))
            self.assertEqual(set(transformers["onehot"].named_steps["encoder"].categories_[0]),
                             {r.weather_main for r in fold.train})

    def test_each_folds_validation_labels_change_scores_but_not_its_predictions(self):
        for index, (start, end) in enumerate(WINDOWS):
            with self.subTest(start=start):
                rows = [
                    {**row, "traffic_volume": row["traffic_volume"] + 100000}
                    if start <= row["date_time"] < end else dict(row)
                    for row in self.rows
                ]
                write_csv(self.path, rows)
                altered = temporal_evaluation.evaluate(self.path)
                original_fold, altered_fold = self.report["folds"][index], altered["folds"][index]
                self.assertEqual([r["predictions"] for r in original_fold["predictions"]],
                                 [r["predictions"] for r in altered_fold["predictions"]])
                self.assertEqual(original_fold["baselines"], altered_fold["baselines"])
                for model in MODEL_NAMES:
                    self.assertNotEqual(original_fold["metrics"][model]["mae"], altered_fold["metrics"][model]["mae"])

    def test_reserved_invalid_values_are_ignored_without_changing_development_results(self):
        rows = [dict(row) for row in self.rows if row["date_time"] < "2018-01-01"]
        rows.extend([
            self.row("2018-01-01 00:00:00", "not a target", "not weather", 999, ""),
            self.row("2018-01-01 00:00:00", "NaN", 0, "invalid", ""),
            self.row("2019-01-01 00:00:00", -1, 0, -1, ""),
        ])
        write_csv(self.path, rows)
        changed = temporal_evaluation.evaluate(self.path)
        self.assertNotEqual(changed["source_sha256"], self.report["source_sha256"])
        for key in ("folds", "pooled_metrics", "paired_mae_differences"):
            self.assertEqual(changed[key], self.report[key])

    def test_conflicting_development_duplicate_fails_before_model_fit(self):
        write_csv(self.path, [*self.rows, {**self.rows[0], "traffic_volume": 99999}])
        with patch("src.temporal_evaluation.make_model") as factory:
            with self.assertRaisesRegex(ValueError, "conflicting traffic_volume"):
                temporal_evaluation.evaluate(self.path)
            factory.assert_not_called()

    def test_evaluation_is_repeatable_without_loading_legacy_artifacts(self):
        with patch.object(pickle, "load", side_effect=AssertionError("Unexpected artifact load")):
            with patch.object(pickle, "loads", side_effect=AssertionError("Unexpected artifact load")):
                self.assertEqual(temporal_evaluation.evaluate(self.path), self.report)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "src.temporal_evaluation", "--data", str(self.path), *args],
            cwd=ROOT, capture_output=True, text=True, check=False, timeout=30,
        )

    def test_cli_stdout_is_only_json(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(result.stdout), self.report)

    def test_cli_writes_new_report_and_refuses_overwrite(self):
        output = Path(self.directory.name) / "report.json"
        first = self.cli("--output", str(output))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, "")
        original = output.read_bytes()
        self.assertEqual(json.loads(original), self.report)
        second = self.cli("--output", str(output))
        self.assertEqual(second.returncode, 2)
        self.assertIn("Output already exists", second.stderr)
        self.assertEqual(second.stdout, "")
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
