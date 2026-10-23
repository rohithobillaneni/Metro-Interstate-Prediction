import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation import baseline_predictions, evaluate, make_model, metrics
from src.offline_data import load_csv, partition_records


ROOT = Path(__file__).resolve().parents[1]


class MetricTests(unittest.TestCase):
    def test_known_mae_rmse_and_r2(self):
        result = metrics([0, 2, 4], [0, 1, 5])
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2 / 3))
        self.assertAlmostEqual(result["r2"], 0.75)

    def test_constant_target_r2_is_null_even_for_perfect_predictions(self):
        for predictions in ([2, 2], [1, 3]):
            with self.subTest(predictions=predictions):
                result = metrics([2, 2], predictions)
                self.assertIsNone(result["r2"])
                self.assertIn('"r2": null', json.dumps(result, allow_nan=False))

    def test_invalid_cohorts_and_nonfinite_values_fail(self):
        for actual, predicted in (
            ([], []), ([1], []), ([1], [1, 2]),
            ([float("nan")], [1]), ([1], [float("inf")]),
            ([float("-inf")], [1]),
        ):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    metrics(actual, predicted)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "source.csv"
        self.rows = []
        for index, (timestamp, target) in enumerate((
            ("2016-12-18 08:00:00", 10),
            ("2016-12-25 08:00:00", 30),
            ("2016-12-19 09:00:00", 50),
            ("2016-12-26 09:00:00", 70),
            ("2016-12-20 10:00:00", 90),
            ("2016-12-27 10:00:00", 110),
        )):
            self.rows.append(self.row(timestamp, target, 273.15 + index, index * 10, "Clouds"))
        self.rows.extend([
            self.row("2017-01-01 08:00:00", 25, 310, 90, "Rain"),
            self.row("2017-01-02 09:00:00", 65, 311, 100, "Rain"),
            self.row("2017-02-03 11:00:00", 100, 312, 95, "Rain"),
            self.row("2018-01-01 08:00:00", 1000, 320, 100, "Snow"),
        ])
        self.write_rows()

    @staticmethod
    def row(timestamp, target, temp, clouds, weather):
        return {
            "date_time": timestamp,
            "traffic_volume": target,
            "temp": temp,
            "clouds_all": clouds,
            "weather_main": weather,
            "holiday": "None",
        }

    def write_rows(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_baselines_use_training_targets_and_global_mean_fallback(self):
        records, _ = load_csv(self.path)
        parts = partition_records(records)
        predictions, fallbacks = baseline_predictions(parts.train, parts.validation)
        self.assertEqual(predictions["training_mean"], [60, 60, 60])
        self.assertEqual(predictions["weekday_hour_mean"], [20, 60, 60])
        self.assertEqual(fallbacks, 1)
        changed_validation = tuple(replace(record, traffic_volume=99999) for record in parts.validation)
        self.assertEqual(baseline_predictions(parts.train, changed_validation), (predictions, fallbacks))
        with self.assertRaisesRegex(ValueError, "nonempty training"):
            baseline_predictions((), parts.validation)

    def test_raw_source_report_reserves_test_and_accounts_for_validation_months(self):
        report = evaluate(self.path)
        self.assertEqual(report["split"]["train"]["count"], 6)
        self.assertEqual(report["split"]["validation"]["count"], 3)
        self.assertEqual(report["split"]["reserved_test"], {
            "count": 1,
            "first_timestamp": "2018-01-01T08:00:00",
            "last_timestamp": "2018-01-01T08:00:00",
            "scored": False,
            "used_for_fit": False,
        })
        self.assertEqual(set(report["metrics"]), {"training_mean", "weekday_hour_mean", "random_forest"})
        self.assertEqual(set(report["validation_months"]), {"1", "2"})
        self.assertEqual(report["validation_months"]["1"]["count"], 2)
        self.assertEqual(report["validation_months"]["2"]["count"], 1)
        self.assertEqual(sum(month["count"] for month in report["validation_months"].values()), 3)
        self.assertEqual(report["weekday_hour_fallback_count"], 1)
        self.assertEqual(report["cleaning"]["raw_count"], 10)
        self.assertEqual(report["cleaning"]["distinct_timestamps"], 10)
        json.dumps(report, allow_nan=False)

    def test_validation_labels_change_scores_without_changing_predictions(self):
        original = evaluate(self.path)
        for row in self.rows:
            if row["date_time"].startswith("2017-"):
                row["traffic_volume"] += 10000
        self.write_rows()
        changed = evaluate(self.path)
        self.assertNotEqual(original["source_sha256"], changed["source_sha256"])
        for model in original["metrics"]:
            with self.subTest(model=model):
                self.assertEqual(
                    original["metrics"][model]["prediction_sha256"],
                    changed["metrics"][model]["prediction_sha256"],
                )
                self.assertNotEqual(original["metrics"][model]["mae"], changed["metrics"][model]["mae"])

    def test_reserved_labels_do_not_change_metrics_or_predictions(self):
        original = evaluate(self.path)
        for row in self.rows:
            if row["date_time"].startswith("2018-"):
                row["traffic_volume"] = 0
        self.write_rows()
        changed = evaluate(self.path)
        self.assertNotEqual(original["source_sha256"], changed["source_sha256"])
        self.assertEqual(original["metrics"], changed["metrics"])
        self.assertEqual(original["validation_months"], changed["validation_months"])
        self.assertEqual(original["split"], changed["split"])

    def test_evaluation_is_repeatable(self):
        self.assertEqual(evaluate(self.path), evaluate(self.path))

    def test_scaler_and_categories_are_fitted_only_on_training_rows(self):
        model = make_model()
        with patch("src.evaluation.make_model", return_value=model):
            evaluate(self.path)
        transformer = model.named_steps["preprocessor"]
        scaler = transformer.named_transformers_["num"].named_steps["scaler"]
        self.assertAlmostEqual(scaler.mean_[0], 2.5)
        self.assertAlmostEqual(scaler.mean_[1], 25)
        self.assertEqual(scaler.n_samples_seen_, 6)
        encoder = transformer.named_transformers_["onehot"].named_steps["encoder"]
        self.assertEqual(encoder.categories_[0].tolist(), ["Clouds"])

    def test_evaluation_does_not_load_legacy_serialized_artifacts(self):
        with patch.object(pickle, "load", side_effect=AssertionError("Unexpected artifact load")):
            with patch.object(pickle, "loads", side_effect=AssertionError("Unexpected artifact load")):
                evaluate(self.path)
        self.assertNotIn("cassandra", sys.modules)
        self.assertNotIn("src.data_preprocessing", sys.modules)
        self.assertNotIn("src.model_training", sys.modules)
        self.assertNotIn("src.model_evaluation", sys.modules)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "src.evaluation", "--data", str(self.path), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_cli_stdout_is_only_json(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        report = json.loads(result.stdout)
        self.assertEqual(report["split"]["validation"]["count"], 3)

    def test_cli_writes_new_output_and_refuses_to_replace_it(self):
        output = Path(self.directory.name) / "report.json"
        first = self.cli("--output", str(output))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, "")
        original = output.read_bytes()
        self.assertEqual(json.loads(original)["schema_version"], 1)
        second = self.cli("--output", str(output))
        self.assertEqual(second.returncode, 2)
        self.assertIn("Output already exists", second.stderr)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(second.stdout, "")


if __name__ == "__main__":
    unittest.main()
