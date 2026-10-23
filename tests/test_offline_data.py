import csv
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from src.offline_data import (
    TrafficRecord,
    encode_holiday,
    feature_values,
    hour_category,
    load_csv,
    partition_records,
)


class FeatureValuesTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            "temp": 12.5,
            "clouds_all": 40,
            "weather_main": "Clouds",
            "holiday": 0,
            "weekday": 5,
            "hour": 23,
            "month": 12,
        }

    def test_feature_values_preserve_valid_inputs_and_add_hour_category(self):
        self.assertEqual(feature_values(**self.values), {
            "temp": 12.5,
            "clouds_all": 40,
            "weather_main": "Clouds",
            "holiday": 0,
            "weekday": 5,
            "hour": 23,
            "month": 12,
            "hour_category": "Night",
        })
        for temp, clouds in ((-273.1499, 0), (0, 100), (20, 12.5)):
            with self.subTest(temp=temp, clouds=clouds):
                actual = feature_values(**{**self.values, "temp": temp, "clouds_all": clouds})
                self.assertEqual(actual["temp"], temp)
                self.assertEqual(actual["clouds_all"], clouds)

    def test_weather_is_trimmed_and_length_bounded_after_trimming(self):
        for raw, expected in ((" \tSnow\n", "Snow"), ("x" * 64, "x" * 64)):
            with self.subTest(raw=raw):
                actual = feature_values(**{**self.values, "weather_main": raw})
                self.assertEqual(actual["weather_main"], expected)
        for value in (None, False, 1, "", " \n\t", "x" * 65):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "weather_main"):
                    feature_values(**{**self.values, "weather_main": value})

    def test_continuous_features_reject_bool_non_numeric_and_nonfinite_values(self):
        for name in ("temp", "clouds_all"):
            for value in (True, False, None, "10", complex(10), float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        feature_values(**{**self.values, name: value})
        with self.assertRaisesRegex(ValueError, "temp"):
            feature_values(**{**self.values, "temp": 10 ** 1000})

    def test_continuous_features_enforce_physical_ranges(self):
        for name, values in (("temp", (-273.15, -274)), ("clouds_all", (-0.1, 100.1))):
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        feature_values(**{**self.values, name: value})

    def test_discrete_features_require_exact_ints_in_range(self):
        ranges = {"holiday": (0, 1), "weekday": (0, 6), "hour": (0, 23), "month": (1, 12)}
        for name, (minimum, maximum) in ranges.items():
            for value in (True, False, None, str(minimum), float(minimum), minimum - 1, maximum + 1):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        feature_values(**{**self.values, name: value})
            for value in (minimum, maximum):
                with self.subTest(name=name, boundary=value):
                    self.assertEqual(feature_values(**{**self.values, name: value})[name], value)

    def test_public_hour_category_handles_boundaries_and_rejects_invalid_hours(self):
        expected = {
            0: "Late Night", 3: "Late Night", 4: "Early Morning", 7: "Early Morning",
            8: "Morning", 11: "Morning", 12: "Afternoon", 15: "Afternoon",
            16: "Evening", 19: "Evening", 20: "Night", 23: "Night",
        }
        for hour, category in expected.items():
            with self.subTest(hour=hour):
                self.assertEqual(hour_category(hour), category)
        for hour in (True, False, None, "1", 1.0, -1, 24):
            with self.subTest(hour=hour):
                with self.assertRaisesRegex(ValueError, "hour"):
                    hour_category(hour)

    def test_feature_results_are_independent_and_records_share_validation(self):
        first = feature_values(**self.values)
        first["weather_main"] = "Rain"
        self.assertEqual(feature_values(**self.values)["weather_main"], "Clouds")
        self.assertEqual(self.values["weather_main"], "Clouds")
        record = TrafficRecord(datetime(2016, 12, 31, 23), 100, 12.5, 40, " Clouds ", 0)
        self.assertEqual(record.features(), feature_values(**self.values))
        with self.assertRaisesRegex(ValueError, "holiday"):
            replace(record, holiday=True).features()


class OfflineDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "traffic.csv"

    def write_rows(self, rows, columns=None):
        columns = columns or [
            "date_time", "holiday", "temp", "clouds_all", "weather_main", "traffic_volume"
        ]
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, date_time="2016-12-31 23:00:00", **changes):
        row = {
            "date_time": date_time,
            "holiday": "None",
            "temp": "273.15",
            "clouds_all": "40",
            "weather_main": "Clouds",
            "traffic_volume": "123",
        }
        row.update(changes)
        return row

    def test_features_convert_kelvin_and_preserve_zero_traffic(self):
        self.write_rows([self.row(traffic_volume="0")])
        records, diagnostics = load_csv(self.path)
        self.assertEqual(records[0].traffic_volume, 0)
        self.assertEqual(records[0].features(), {
            "temp": 0, "clouds_all": 40, "weather_main": "Clouds", "holiday": 0,
            "weekday": 5, "hour": 23, "month": 12, "hour_category": "Night",
        })
        self.assertEqual(diagnostics.raw_count, 1)
        self.assertEqual(diagnostics.distinct_timestamps, 1)
        with self.assertRaises(FrozenInstanceError):
            records[0].temp = 3
        features = records[0].features()
        features["temp"] = 999
        self.assertEqual(records[0].features()["temp"], 0)

    def test_holiday_encoding_handles_source_null_and_named_values(self):
        for value in (None, "", " ", "None", " None "):
            with self.subTest(value=value):
                self.assertEqual(encode_holiday(value), 0)
        self.assertEqual(encode_holiday("Christmas Day"), 1)
        self.write_rows([self.row(holiday="Christmas Day")])
        self.assertEqual(load_csv(self.path)[0][0].holiday, 1)

    def test_hour_category_boundaries(self):
        expected = {
            0: "Late Night", 3: "Late Night", 4: "Early Morning", 7: "Early Morning",
            8: "Morning", 11: "Morning", 12: "Afternoon", 15: "Afternoon",
            16: "Evening", 19: "Evening", 20: "Night", 23: "Night",
        }
        self.write_rows([
            self.row(date_time=f"2016-12-31 {hour:02d}:00:00") for hour in expected
        ])
        records, _ = load_csv(self.path)
        self.assertEqual(
            {record.date_time.hour: record.features()["hour_category"] for record in records},
            expected,
        )

    def test_duplicate_weather_keeps_first_source_row_and_sorts_by_timestamp(self):
        self.write_rows([
            self.row("2017-01-01 00:00:00", weather_main="Rain", temp="290"),
            self.row("2016-12-31 23:00:00"),
            self.row("2017-01-01 00:00:00", weather_main="Snow", temp="270"),
        ])
        records, diagnostics = load_csv(self.path)
        self.assertEqual([record.date_time.year for record in records], [2016, 2017])
        self.assertEqual(records[1].weather_main, "Rain")
        self.assertAlmostEqual(records[1].temp, 16.85)
        self.assertEqual(diagnostics.raw_count, 3)
        self.assertEqual(diagnostics.duplicate_extras, 1)
        self.assertEqual(diagnostics.distinct_timestamps, 2)

    def test_conflicting_targets_fail_even_when_conflict_has_invalid_temperature(self):
        for first_temp, second_temp in (("273", "274"), ("0", "274"), ("273", "0")):
            with self.subTest(first=first_temp, second=second_temp):
                self.write_rows([
                    self.row(temp=first_temp, traffic_volume="1"),
                    self.row(temp=second_temp, traffic_volume="2"),
                ])
                with self.assertRaisesRegex(ValueError, "conflicting traffic_volume"):
                    load_csv(self.path)

    def test_nonpositive_kelvin_exclusion_has_explicit_counts(self):
        self.write_rows([
            self.row(temp="0"),
            self.row(temp="-1"),
            self.row(temp="280"),
            self.row(temp="290"),
            self.row("2017-01-01 00:00:00", temp="310", traffic_volume="0"),
        ])
        records, diagnostics = load_csv(self.path)
        self.assertEqual(diagnostics.invalid_temp_dropped, 2)
        self.assertEqual(diagnostics.duplicate_extras, 1)
        self.assertEqual(diagnostics.raw_count, 5)
        self.assertEqual(diagnostics.distinct_timestamps, 2)
        self.assertAlmostEqual(records[0].temp, 6.85)
        self.assertEqual(records[1].traffic_volume, 0)

    def test_malformed_and_nonfinite_numeric_values_fail(self):
        for column in ("traffic_volume", "temp", "clouds_all"):
            for value in ("bad", "", "nan", "inf", "-inf"):
                with self.subTest(column=column, value=value):
                    self.write_rows([self.row(**{column: value})])
                    with self.assertRaisesRegex(ValueError, column):
                        load_csv(self.path)
        self.write_rows([self.row(traffic_volume="-1")])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_csv(self.path)
        for clouds in ("-1", "101"):
            with self.subTest(clouds=clouds):
                self.write_rows([self.row(clouds_all=clouds)])
                with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                    load_csv(self.path)

    def test_invalid_timestamp_and_missing_weather_fail(self):
        for value in ("not a date", "2017-02-30 00:00:00", "2017-01-01", "2017-01-01T00:00:00Z"):
            with self.subTest(value=value):
                self.write_rows([self.row(date_time=value)])
                with self.assertRaisesRegex(ValueError, "date_time"):
                    load_csv(self.path)
        self.write_rows([self.row(weather_main=" ")])
        with self.assertRaisesRegex(ValueError, "weather_main"):
            load_csv(self.path)

    def test_header_and_row_shape_errors_fail(self):
        for content, message in (
            ("", "no header"),
            ("date_time,temp\n", "missing required columns"),
            ("temp,temp\n", "duplicate column names"),
            ("date_time,holiday,temp,clouds_all,weather_main,traffic_volume\nx,\n", "row width"),
            ("date_time,holiday,temp,clouds_all,weather_main,traffic_volume\nx,x,x,x,x,x,extra\n", "row width"),
        ):
            with self.subTest(message=message):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_csv(self.path)

    def test_empty_and_fully_excluded_data_fail(self):
        for rows in ([], [self.row(temp="0")]):
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, "no usable records"):
                    load_csv(self.path)

    def test_calendar_partitions_respect_exact_boundaries_and_sort_input(self):
        self.write_rows([
            self.row("2018-01-01 00:00:00", traffic_volume="999"),
            self.row("2017-12-31 23:00:00"),
            self.row("2016-12-31 23:00:00"),
            self.row("2017-01-01 00:00:00"),
        ])
        records, _ = load_csv(self.path)
        partitions = partition_records(reversed(records))
        self.assertEqual(len(partitions.train), 1)
        self.assertEqual([record.date_time for record in partitions.validation], [
            datetime(2017, 1, 1), datetime(2017, 12, 31, 23)
        ])
        self.assertEqual(partitions.test[0].date_time, datetime(2018, 1, 1))
        changed = [replace(record, traffic_volume=0) for record in records]
        changed_partitions = partition_records(changed)
        self.assertEqual(
            [record.date_time for record in changed_partitions.train],
            [record.date_time for record in partitions.train],
        )

    def test_invalid_partitions_fail_and_test_can_be_empty(self):
        self.write_rows([self.row(), self.row("2017-01-01 00:00:00")])
        records, _ = load_csv(self.path)
        self.assertEqual(partition_records(records).test, ())
        for validation, test in (
            ("2018-01-01", "2017-01-01"),
            ("2017-01-01", "2017-01-01"),
            ("invalid", "2018-01-01"),
            ("2017-01-01T00:00:00+00:00", "2018-01-01"),
        ):
            with self.subTest(validation=validation, test=test):
                with self.assertRaises(ValueError):
                    partition_records(records, validation, test)
        for subset in ([], records[:1], records[1:]):
            with self.subTest(subset=subset):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    partition_records(subset)
        with self.assertRaisesRegex(ValueError, "unique timestamps"):
            partition_records(records + records)
        with self.assertRaisesRegex(ValueError, "naive"):
            partition_records([
                records[0],
                replace(records[1], date_time=records[1].date_time.replace(tzinfo=timezone.utc)),
            ])


if __name__ == "__main__":
    unittest.main()
