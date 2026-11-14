import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from src import predict
from src.app import create_app
from src.artifacts import load_bundle, train_bundle


ROOT = Path(__file__).resolve().parents[1]
REQUEST = {
    "temp": 15, "clouds_all": 50, "holiday": "No", "weather_main": "Clouds",
    "weekday": "Monday", "hour": 12, "month": "December",
}


class BatchPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        root = Path(cls.fixture.name)
        source = root / "training.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("date_time", "traffic_volume", "temp", "clouds_all", "weather_main", "holiday"))
            for index in range(6):
                writer.writerow((f"2016-12-{20 + index} 12:00:00", 100 * index, 280 + index, 10 * index, "Clouds", "None"))
            writer.writerow(("2017-01-01 12:00:00", 20, 285, 20, "Clear", "None"))
        cls.bundle = root / "bundle"
        cls.pin = train_bundle(source, cls.bundle)
        cls.model = load_bundle(cls.bundle, cls.pin)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "requests.json"
        self.output = self.root / "predictions.json"
        self.requests = [REQUEST, {**REQUEST, "temp": -5, "hour": 23, "weather_main": " Unseen weather "}, REQUEST]
        self.source.write_text(json.dumps(self.requests), encoding="utf-8")

    def arguments(self):
        return ["--input", str(self.source), "--model-dir", str(self.bundle), "--manifest-sha256", self.pin]

    def assert_cli_failure(self, message):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                predict.main([*self.arguments(), "--output", str(self.output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse(self.output.exists())
        return stderr.getvalue()

    def test_batch_matches_api_in_input_order_and_records_exact_hashes(self):
        encoded = self.source.read_bytes()
        report = predict.predict_file(self.source, self.bundle, self.pin)
        with TestClient(create_app(self.bundle, self.pin)) as client:
            expected = []
            for request in self.requests:
                response = client.post("/predict", json=request)
                self.assertEqual(response.status_code, 200)
                expected.append(response.json()["prediction"])
        self.assertEqual(report["predictions"], [
            {"row": index, "prediction": value} for index, value in enumerate(expected, start=1)
        ])
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["input_sha256"], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(report["manifest_sha256"], self.pin)
        self.assertEqual(report["schema_version"], 1)
        self.assertNotIn("metrics", report)
        self.assertNotIn("requests", report)

    def test_invalid_later_request_prevents_model_loading_and_echo(self):
        invalid = (
            {**REQUEST, "temp": True}, {**REQUEST, "temp": "15"},
            {**REQUEST, "hour_category": "Morning"}, {**REQUEST, "traffic_volume": 500},
            {**REQUEST, "weather_main": "PRIVATE-INPUT-" * 10},
            {name: value for name, value in REQUEST.items() if name != "month"},
            None, [], "PRIVATE-INPUT",
        )
        for request in invalid:
            with self.subTest(request=request), patch.object(predict, "load_bundle") as load:
                self.source.write_text(json.dumps([REQUEST, request]))
                error = self.assert_cli_failure("Request 2 does not match")
                self.assertNotIn("PRIVATE-INPUT", error)
                load.assert_not_called()

    def test_malformed_json_utf8_duplicate_keys_and_empty_batches_fail_closed(self):
        invalid = (
            b"", b"[", b"\xff", b"[]", b"{}", b"null", b"[NaN]", b"[Infinity]",
            b'[{"temp": 1, "temp": 2}]', b"[" * 1500 + b"]" * 1500,
        )
        for encoded in invalid:
            with self.subTest(encoded=encoded[:30]), patch.object(predict, "load_bundle") as load:
                self.source.write_bytes(encoded)
                self.assert_cli_failure("error:")
                load.assert_not_called()

    def test_byte_and_request_limits_apply_before_model_loading(self):
        for limit, value in (("MAX_INPUT_BYTES", self.source.stat().st_size - 1), ("MAX_REQUESTS", 2)):
            with self.subTest(limit=limit), patch.object(predict, limit, value), patch.object(predict, "load_bundle") as load:
                self.assert_cli_failure("limit" if limit == "MAX_INPUT_BYTES" else "1 to 2")
                load.assert_not_called()
        with patch.object(predict, "MAX_INPUT_BYTES", self.source.stat().st_size), patch.object(predict, "MAX_REQUESTS", 3):
            self.assertEqual(predict.predict_file(self.source, self.bundle, self.pin)["count"], 3)

    def test_wrong_pin_is_rejected_before_deserialization(self):
        with patch("src.artifacts.pickle.loads") as loads:
            with self.assertRaisesRegex(ValueError, "separately trusted pin"):
                predict.predict_file(self.source, self.bundle, "0" * 64)
            loads.assert_not_called()

    def test_invalid_model_output_produces_no_report(self):
        invalid = ([], [1, 2], [[1], [2], [3]], 1, [1, -1, 3], [1, float("nan"), 3],
                   [1, float("inf"), 3], [True, False, True], ["1", "2", "3"])
        for values in invalid:
            with self.subTest(values=values), patch.object(predict, "load_bundle", return_value=self.model):
                with patch.object(self.model, "predict", return_value=np.asarray(values)):
                    self.assert_cli_failure("Model")

    def test_zero_prediction_is_retained(self):
        with patch.object(predict, "load_bundle", return_value=self.model):
            with patch.object(self.model, "predict", return_value=np.zeros(3)):
                report = predict.predict_file(self.source, self.bundle, self.pin)
        self.assertEqual(report["predictions"], [{"row": row, "prediction": 0.0} for row in (1, 2, 3)])

    def test_input_hash_binds_parsed_bytes_even_if_path_changes_later(self):
        original = self.source.read_bytes()
        def load(*args):
            self.source.write_bytes(b"changed after parsing")
            return self.model
        with patch.object(predict, "load_bundle", side_effect=load):
            report = predict.predict_file(self.source, self.bundle, self.pin)
        self.assertEqual(report["input_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(report["count"], 3)

    def test_existing_output_is_preserved_without_loading_model(self):
        self.output.write_bytes(b"existing evidence")
        with patch.object(predict, "load_bundle") as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                predict.main([*self.arguments(), "--output", str(self.output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.output.read_bytes(), b"existing evidence")
        load.assert_not_called()

    def test_output_created_during_prediction_is_not_overwritten(self):
        original = self.model.predict
        def concurrent_output(features):
            self.output.write_bytes(b"another writer's evidence")
            return original(features)
        with patch.object(predict, "load_bundle", return_value=self.model), patch.object(self.model, "predict", side_effect=concurrent_output):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    predict.main([*self.arguments(), "--output", str(self.output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.output.read_bytes(), b"another writer's evidence")

    def test_module_cli_writes_report_to_new_file_or_stdout(self):
        command = [sys.executable, "-m", "src.predict", *self.arguments()]
        result = subprocess.run([*command, "--output", str(self.output)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        report = json.loads(self.output.read_text())
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), report)


if __name__ == "__main__":
    unittest.main()
