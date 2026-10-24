import csv
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import temporal_evaluation, weather_sensitivity
from src.evaluation import make_model
from src.offline_data import load_csv, load_development_csv


def row(stamp, target=100, weather="Clouds", temp=290):
    return dict(date_time=stamp, traffic_volume=target, temp=temp, clouds_all=30,
                weather_main=weather, holiday="None")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class WeatherPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.csv"

    def test_first_last_select_valid_rows_and_preserve_zero_and_counts(self):
        write_csv(self.path, [row("2017-01-01 00:00:00", 0, "bad", 0),
                              row("2017-01-01 00:00:00", 0, "Clouds"),
                              row("2017-01-01 00:00:00", 0, "Rain"),
                              row("2017-01-01 00:00:00", 0, "bad", 0)])
        first, diagnostics = load_development_csv(self.path)
        last, other = load_development_csv(self.path, weather_row="last")
        self.assertEqual(first[0].weather_main, "Clouds")
        self.assertEqual(last[0].weather_main, "Rain")
        self.assertEqual(last[0].traffic_volume, 0)
        self.assertEqual(first, load_csv(self.path)[0])
        self.assertEqual(diagnostics, other)
        self.assertEqual((other.raw_count, other.invalid_temp_dropped,
                          other.duplicate_extras, other.distinct_timestamps), (4, 2, 1, 1))

    def test_conflicting_target_even_in_invalid_temperature_row_fails_both_policies(self):
        write_csv(self.path, [row("2017-01-01 00:00:00"),
                              row("2017-01-01 00:00:00", 101, temp=0)])
        for policy in ("first", "last"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "conflicting"):
                load_development_csv(self.path, weather_row=policy)

    def test_reserved_values_do_not_affect_either_policy(self):
        rows = [row("2017-01-01 00:00:00"), row("2017-01-01 00:00:00", weather="Rain")]
        write_csv(self.path, rows)
        expected = {p: load_development_csv(self.path, weather_row=p)[0] for p in ("first", "last")}
        rows.extend([row("2018-01-01 00:00:00", "secret-not-a-number", "", "broken"),
                     row("2018-01-01 00:00:00", -10, "", "nan")])
        write_csv(self.path, rows)
        for policy in ("first", "last"):
            actual, counts = load_development_csv(self.path, weather_row=policy)
            self.assertEqual(actual, expected[policy])
            self.assertEqual(counts.reserved_rows_excluded, 2)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "weather_row"):
            load_development_csv(self.path, weather_row="best")


class FullYearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "data.csv"
        cls.rows = [row("2016-12-25 00:00:00", 0), row("2016-12-31 23:00:00", 200)]
        for index, (start, end) in enumerate(weather_sensitivity.WINDOWS):
            for hour in range(index % 3 + 1):
                stamp = (datetime.fromisoformat(start) + timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")
                cls.rows.extend([row(stamp, 100 + index * 5 + hour),
                                 row(stamp, 100 + index * 5 + hour, "Rain", 280)])
        cls.rows.append(row("2018-01-01 00:00:00", "untouched", "", "invalid"))
        write_csv(cls.path, cls.rows)
        cls.models = []
        def create():
            model = make_model()
            cls.models.append(model)
            return model
        with patch.object(temporal_evaluation, "make_model", side_effect=create):
            cls.report = weather_sensitivity.evaluate(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_full_year_windows_exhaustive_disjoint_and_last_day_preserved(self):
        records, _ = load_development_csv(self.path)
        folds = weather_sensitivity.full_year_folds(tuple(reversed(records)))
        self.assertEqual(len(folds), 53)
        self.assertEqual(folds[0].start, datetime(2017, 1, 1))
        self.assertEqual(folds[-1].start, datetime(2017, 12, 31))
        self.assertEqual(folds[-1].end, datetime(2018, 1, 1))
        actual = [r.date_time for f in folds for r in f.validation]
        self.assertEqual(actual, [r.date_time for r in records if r.date_time.year == 2017])
        for index, fold in enumerate(folds):
            self.assertLess(max(r.date_time for r in fold.train), fold.start)
            self.assertLessEqual(fold.end - fold.start, timedelta(days=7))
            self.assertTrue(all(r.date_time - timedelta(days=7) < fold.start for r in fold.validation))
            self.assertEqual(fold.end, folds[index+1].start if index < 52 else datetime(2018,1,1))

    def test_empty_window_rejected_without_shifting_boundaries(self):
        records, _ = load_development_csv(self.path)
        records = tuple(r for r in records if not datetime(2017,5,7) <= r.date_time < datetime(2017,5,14))
        with self.assertRaises(ValueError):
            weather_sensitivity.full_year_folds(records)

    def test_baselines_identical_and_each_pipeline_is_fresh(self):
        self.assertEqual(len(self.models), 106)
        self.assertEqual(len({id(model) for model in self.models}), 106)
        for first, last in zip(self.report["policies"]["first"]["folds"], self.report["policies"]["last"]["folds"]):
            for key in ("train", "validation", "coverage", "baselines"):
                self.assertEqual(first[key], last[key])
            for a, b in zip(first["predictions"], last["predictions"]):
                self.assertEqual((a["date_time"],a["actual"]), (b["date_time"],b["actual"]))
                for name in temporal_evaluation.BASELINES:
                    self.assertEqual(a["predictions"][name], b["predictions"][name])
        for name in temporal_evaluation.BASELINES:
            self.assertEqual(self.report["last_minus_first_metrics"][name], {"mae":0, "rmse":0})

    def test_counts_changed_features_coverage_and_partial_window(self):
        count = sum(index % 3 + 1 for index in range(53))
        self.assertEqual(self.report["feature_changes"]["validation_2017"], {
            "count":count, "changed_rows":count,
            "changed_fields":{"temp":count,"clouds_all":0,"weather_main":count,"holiday":0},
        })
        for policy in self.report["policies"].values():
            self.assertEqual(policy["coverage"]["expected_clock_hours"],8760)
            self.assertEqual(policy["coverage"]["observed_hours"],count)
            self.assertEqual(policy["coverage"]["missing_clock_hours"],8760-count)
            self.assertEqual(policy["folds"][-1]["coverage"]["validation"]["expected_clock_hours"],24)
            self.assertEqual(policy["pooled"]["count"],count)

    def test_pooled_and_quarter_metrics_recomputed_from_every_residual(self):
        for policy in self.report["policies"].values():
            all_rows = [r for fold in policy["folds"] for r in fold["predictions"]]
            cohorts = [(all_rows,policy["pooled"])]
            cohorts.extend(([r for r in all_rows if (datetime.fromisoformat(r["date_time"]).month-1)//3+1==q],policy["quarters"][f"Q{q}"]) for q in range(1,5))
            self.assertEqual(sum(c["count"] for c in policy["quarters"].values()),len(all_rows))
            for rows, result in cohorts:
                self.assertEqual(result["count"],len(rows))
                for name, scores in result["metrics"].items():
                    residuals=[r["actual"]-r["predictions"][name] for r in rows]
                    mae=math.fsum(abs(v) for v in residuals)/len(rows)
                    rmse=math.sqrt(math.fsum(v*v for v in residuals)/len(rows))
                    self.assertAlmostEqual(scores["mae"],mae)
                    self.assertAlmostEqual(scores["rmse"],rmse)
            for baseline,difference in policy["pooled"]["paired_mae_differences"].items():
                scores=policy["pooled"]["metrics"]
                self.assertEqual(difference,scores["random_forest"]["mae"]-scores[baseline]["mae"])
        for name, deltas in self.report["last_minus_first_metrics"].items():
            for metric,value in deltas.items():
                self.assertEqual(value,self.report["policies"]["last"]["pooled"]["metrics"][name][metric]-self.report["policies"]["first"]["pooled"]["metrics"][name][metric])

    def test_cohort_and_baseline_mismatch_fail(self):
        first,_=load_development_csv(self.path)
        for altered in (first[:-1], tuple(replace(r,traffic_volume=r.traffic_volume+1) for r in first)):
            with self.assertRaisesRegex(ValueError,"identical"):
                weather_sensitivity.feature_changes(first,altered)
        a=self.report["policies"]["first"]["folds"][0]["predictions"]
        b=json.loads(json.dumps(a));b[0]["actual"]+=1
        with self.assertRaisesRegex(ValueError,"identical"):
            weather_sensitivity.paired_differences(a,b)
        b=json.loads(json.dumps(a));b[0]["predictions"]["seasonal_7_day"]+=1
        with self.assertRaisesRegex(ValueError,"baselines"):
            weather_sensitivity.paired_differences(a,b)

    def test_saved_protocol_and_source_provenance(self):
        protocol=self.report["protocol"]
        self.assertFalse(protocol["reserved_scored"])
        self.assertFalse(protocol["reserved_targets_validated"])
        self.assertEqual(json.loads(weather_sensitivity.PROTOCOL_PATH.read_bytes()),protocol)
        self.assertEqual(self.report["protocol_sha256"],hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(self.report["protocol_file_sha256"],hashlib.sha256(weather_sensitivity.PROTOCOL_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.report["source_code_sha256"],weather_sensitivity.source_hashes())
        self.assertEqual(self.report["source_sha256"],hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_last_window_labels_do_not_change_predictions_for_either_policy(self):
        changed=[{**r,"traffic_volume":float(r["traffic_volume"])+10000} if "2017-12-31"<=r["date_time"]<"2018" else r for r in self.rows]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"changed.csv";write_csv(path,changed)
            report=weather_sensitivity.evaluate(path)
        for name in ("first","last"):
            for actual,expected in zip(report["policies"][name]["folds"],self.report["policies"][name]["folds"]):
                self.assertEqual([r["predictions"] for r in actual["predictions"]],[r["predictions"] for r in expected["predictions"]])
            self.assertNotEqual(report["policies"][name]["pooled"]["metrics"],self.report["policies"][name]["pooled"]["metrics"])

    def test_repeat_and_reserved_mutation_leave_all_development_results_unchanged(self):
        self.assertEqual(weather_sensitivity.evaluate(self.path),self.report)
        changed=[r if r["date_time"]<"2018" else row(r["date_time"],-500,"private-not-used","nan") for r in self.rows]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"changed.csv";write_csv(path,changed)
            report=weather_sensitivity.evaluate(path)
        self.assertEqual(report["policies"],self.report["policies"])
        self.assertEqual(report["feature_changes"],self.report["feature_changes"])
        self.assertNotEqual(report["source_sha256"],self.report["source_sha256"])

    def test_cli_refuses_overwrite_before_evaluation(self):
        with patch.object(weather_sensitivity,"evaluate") as evaluate:
            with self.assertRaises(SystemExit):
                weather_sensitivity.main(["--data",str(self.path),"--output",str(self.path)])
            evaluate.assert_not_called()

    def test_protocol_drift_fails_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"protocol.json";path.write_text('{}')
            with patch.object(weather_sensitivity,"PROTOCOL_PATH",path), patch.object(temporal_evaluation,"make_model") as model:
                with self.assertRaisesRegex(ValueError,"saved experiment protocol"):
                    weather_sensitivity.evaluate(self.path)
                model.assert_not_called()


class WindowContractTests(unittest.TestCase):
    def test_invalid_window_geometry_rejected(self):
        from src.offline_data import TrafficRecord
        records = (TrafficRecord(datetime(2016,12,31),100,10,30,"Rain",0),
                   TrafficRecord(datetime(2017,1,1),100,10,30,"Rain",0))
        for windows in (
            (("2017-01-01", "2017-01-09"),),
            (("2017-01-01", "2017-01-01"),),
            (("2017-01-01T00:30:00", "2017-01-02"),),
            (("2017-01-01T00:00:00+01:00", "2017-01-02"),),
            (("2017-12-31", "2018-01-02"),),
            (("2017-01-01", "2017-01-08"),("2017-01-07", "2017-01-10")),
        ):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                temporal_evaluation.folds_for_windows(records,windows)
