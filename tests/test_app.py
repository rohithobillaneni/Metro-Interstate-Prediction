from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from src.app import TrafficData, app, create_app
from src.artifacts import load_bundle, train_bundle


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.source = Path(cls.directory.name) / "source.csv"
        cls.bundle = Path(cls.directory.name) / "model"
        rows = []
        for index in range(6):
            rows.append({
                "date_time": f"2016-12-{index + 20:02d} 08:00:00",
                "holiday": "None",
                "temp": 273.15 + index,
                "clouds_all": index * 10,
                "weather_main": "Clouds",
                "traffic_volume": (index + 1) * 10,
            })
        rows.extend([
            {**rows[0], "date_time": "2017-01-01 08:00:00", "traffic_volume": 1000},
            {**rows[0], "date_time": "2018-01-01 08:00:00", "traffic_volume": 2000},
        ])
        with cls.source.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cls.pin = train_bundle(cls.source, cls.bundle)
        cls.direct_model = load_bundle(cls.bundle, cls.pin)

    def setUp(self):
        self.payload = {
            "temp": 2.0,
            "clouds_all": 20.0,
            "holiday": "No",
            "weather_main": "Clouds",
            "weekday": "Tuesday",
            "hour": 8,
            "month": "December",
        }

    def client(self):
        return TestClient(create_app(self.bundle, self.pin))

    def test_import_requires_no_artifact_configuration_or_deserialization(self):
        environment = os.environ.copy()
        environment.pop("METRO_MODEL_DIR", None)
        environment.pop("METRO_MODEL_MANIFEST_SHA256", None)
        environment["PYTHONPATH"] = str(ROOT)
        script = """
import pickle
import socket
from unittest.mock import patch
with patch.object(pickle, 'load', side_effect=AssertionError('Unexpected artifact load')):
    with patch.object(pickle, 'loads', side_effect=AssertionError('Unexpected artifact load')):
        with patch.object(socket, 'create_connection', side_effect=AssertionError('Unexpected network')):
            import src.app
            assert not hasattr(src.app.app.state, 'model')
print('imported without a model')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.directory.name,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "imported without a model\n")
        self.assertEqual(result.stderr, "")

    def test_lifespan_fails_without_both_configuration_values(self):
        with patch.dict(os.environ, {}, clear=True):
            for directory, pin in ((None, None), (self.bundle, None), (None, self.pin)):
                with self.subTest(directory=directory, pin=pin):
                    with self.assertRaisesRegex(RuntimeError, "separately trusted"):
                        with TestClient(create_app(directory, pin)):
                            self.fail("Unconfigured application started")

    def test_lifespan_rejects_missing_bundle_and_wrong_manifest_pin(self):
        for directory, pin, error in (
            (Path(self.directory.name) / "missing", self.pin, "regular non-symlink"),
            (self.bundle, "0" * 64, "separately trusted pin"),
            (self.bundle, "bad-pin", "SHA256 manifest pin"),
        ):
            with self.subTest(directory=directory, pin=pin):
                with self.assertRaisesRegex(ValueError, error):
                    with TestClient(create_app(directory, pin)):
                        self.fail("Unverified application started")

    def test_root_and_prediction_match_direct_fresh_model(self):
        application = create_app(self.bundle, self.pin)
        with TestClient(application) as client:
            self.assertEqual(client.get("/").json(), {
                "message": "Welcome to the Metro Traffic Volume Prediction API!"
            })
            response = client.post("/predict", json=self.payload)
            self.assertEqual(response.status_code, 200, response.text)
            expected = self.direct_model.predict(pd.DataFrame([TrafficData(**self.payload).features()]))[0]
            self.assertEqual(response.json(), {"prediction": float(expected)})
            self.assertTrue(hasattr(application.state, "model"))
        self.assertFalse(hasattr(application.state, "model"))

    def test_default_application_uses_configured_bundle_environment(self):
        with patch.dict(os.environ, {
            "METRO_MODEL_DIR": str(self.bundle),
            "METRO_MODEL_MANIFEST_SHA256": self.pin,
        }):
            with TestClient(app) as client:
                self.assertEqual(client.get("/").status_code, 200)
                response = client.post("/predict", json=self.payload)
                self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(hasattr(app.state, "model"))

    def test_negative_celsius_and_zero_numeric_inputs_are_valid(self):
        with self.client() as client:
            for changes in (
                {"temp": -20.0},
                {"temp": 0, "clouds_all": 0, "hour": 0, "weekday": "Monday", "month": "January"},
            ):
                with self.subTest(changes=changes):
                    response = client.post("/predict", json={**self.payload, **changes})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertGreaterEqual(response.json()["prediction"], 0)

    def test_optional_category_is_derived_and_contradictions_are_rejected(self):
        with self.client() as client:
            derived = client.post("/predict", json=self.payload)
            explicit = client.post("/predict", json={**self.payload, "hour_category": "Morning"})
            self.assertEqual(derived.status_code, 200)
            self.assertEqual(explicit.status_code, 200)
            self.assertEqual(derived.json(), explicit.json())
            response = client.post("/predict", json={**self.payload, "hour_category": "Night"})
            self.assertEqual(response.status_code, 422)
            self.assertIn("contradicts hour", response.text)

    def test_numeric_values_are_strict_and_range_checked(self):
        invalid = {
            "temp": (True, False, "2", None, -273.15, -274),
            "clouds_all": (True, False, "20", None, -0.1, 100.1),
            "hour": (True, False, "8", 8.0, None, -1, 24),
        }
        with self.client() as client:
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value):
                        response = client.post("/predict", json={**self.payload, field: value})
                        self.assertEqual(response.status_code, 422, response.text)

    def test_categorical_values_and_missing_or_extra_fields_are_rejected(self):
        invalid = {
            "holiday": (None, False, 0, "None", "Maybe"),
            "weekday": (None, 0, "monday", "Funday"),
            "month": (None, 1, "january", "Smarch"),
            "weather_main": (None, False, 1, "", " \t\n", "x" * 65),
        }
        with self.client() as client:
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value):
                        response = client.post("/predict", json={**self.payload, field: value})
                        self.assertEqual(response.status_code, 422, response.text)
            for field in self.payload:
                with self.subTest(missing=field):
                    incomplete = {name: value for name, value in self.payload.items() if name != field}
                    response = client.post("/predict", json=incomplete)
                    self.assertEqual(response.status_code, 422, response.text)
            response = client.post("/predict", json={**self.payload, "debug": True})
            self.assertEqual(response.status_code, 422)

    def test_nonfinite_json_returns_serializable_errors_without_echoing_input(self):
        sentinel = "DO-NOT-ECHO-REQUEST-DATA"
        with self.client() as client:
            for field in ("temp", "clouds_all"):
                for value in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(field=field, value=value):
                        response = client.post(
                            "/predict",
                            content=json.dumps({**self.payload, "weather_main": sentinel, field: value}),
                            headers={"Content-Type": "application/json"},
                        )
                        self.assertEqual(response.status_code, 422, response.text)
                        self.assertNotIn(sentinel, response.text)
                        self.assertNotIn("NaN", response.text)
                        self.assertNotIn("Infinity", response.text)
                        for detail in response.json()["detail"]:
                            self.assertEqual(set(detail), {"loc", "type", "msg"})

    def test_unknown_weather_is_supported_by_the_fitted_transformer(self):
        payload = {**self.payload, "weather_main": " Unseen weather "}
        expected = self.direct_model.predict(pd.DataFrame([TrafficData(**payload).features()]))[0]
        with self.client() as client:
            response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"prediction": float(expected)})

    def test_prediction_failures_have_generic_responses(self):
        with self.client() as client:
            with patch.object(client.app.state.model, "predict", side_effect=RuntimeError("sensitive-model-path")):
                response = client.post("/predict", json=self.payload)
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"detail": "Prediction failed"})
            for value in (float("nan"), float("inf"), -1):
                with self.subTest(prediction=value):
                    with patch.object(client.app.state.model, "predict", return_value=[value]):
                        response = client.post("/predict", json=self.payload)
                    self.assertEqual(response.status_code, 500)
                    self.assertEqual(response.json(), {"detail": "Prediction failed"})

    def test_concurrent_requests_match_serial_prediction(self):
        with self.client() as client:
            expected = client.post("/predict", json=self.payload).json()

            def request(_):
                response = client.post("/predict", json=self.payload)
                return response.status_code, response.json()

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(request, range(12)))
        self.assertEqual(results, [(200, expected)] * 12)


if __name__ == "__main__":
    unittest.main()
