import contextlib
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.pipeline import Pipeline

from src import artifacts
from src.evaluation import make_model
from src.offline_data import load_csv


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture_directory.cleanup)
        cls.fixture_root = Path(cls.fixture_directory.name)
        cls.source = cls.fixture_root / "source.csv"
        cls.rows = []
        for index, (timestamp, target) in enumerate((
            ("2016-12-18 08:00:00", 10),
            ("2016-12-19 09:00:00", 50),
            ("2016-12-20 10:00:00", 90),
            ("2016-12-25 08:00:00", 30),
            ("2016-12-26 09:00:00", 70),
            ("2016-12-31 23:00:00", 110),
            ("2017-01-01 00:00:00", 25),
            ("2017-01-02 09:00:00", 65),
            ("2017-12-31 23:00:00", 100),
            ("2018-01-01 00:00:00", 1000),
        )):
            cls.rows.append({
                "date_time": timestamp,
                "traffic_volume": target,
                "temp": 273.15 + index,
                "clouds_all": index * 10,
                "weather_main": "Clouds" if index < 6 else "Rain",
                "holiday": "None",
            })
        cls.rows.append({**cls.rows[0], "weather_main": "Mist"})
        write_csv(cls.source, cls.rows)
        records, _ = load_csv(cls.source)
        cls.features = pd.DataFrame([record.features() for record in records])
        cls.fitted_model = make_model()
        cls.fixture_bundle = cls.fixture_root / "bundle"
        with patch("src.artifacts.make_model", return_value=cls.fitted_model):
            cls.pin = artifacts.train_bundle(cls.source, cls.fixture_bundle)
        cls.original_manifest = (cls.fixture_bundle / "manifest.json").read_bytes()
        cls.original_payload = (cls.fixture_bundle / "model.pkl").read_bytes()
        cls.expected_predictions = cls.fitted_model.predict(cls.features).tolist()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bundle = self.root / "bundle"
        shutil.copytree(self.fixture_bundle, self.bundle)

    def rewrite_manifest(self, manifest):
        encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        (self.bundle / "manifest.json").write_bytes(encoded)
        return digest(encoded)

    def manifest(self):
        return json.loads(self.original_manifest)

    def assert_rejected_before_unpickling(self, pin=None, directory=None, message=None):
        pin = self.pin if pin is None else pin
        directory = self.bundle if directory is None else directory
        with patch("src.artifacts.pickle.loads") as loads:
            if message is None:
                with self.assertRaises(ValueError):
                    artifacts.load_bundle(directory, pin)
            else:
                with self.assertRaisesRegex(ValueError, message):
                    artifacts.load_bundle(directory, pin)
            loads.assert_not_called()

    def test_fresh_bundle_roundtrip_preserves_predictions_and_exact_payload(self):
        self.assertEqual(self.pin, digest(self.original_manifest))
        self.assertEqual({p.name for p in self.bundle.iterdir()}, {"model.pkl", "manifest.json"})
        with patch("src.artifacts.pickle.loads", wraps=pickle.loads) as loads:
            restored = artifacts.load_bundle(self.bundle, self.pin)
        loads.assert_called_once_with(self.original_payload)
        self.assertIsInstance(restored, Pipeline)
        self.assertEqual(restored.predict(self.features).tolist(), self.expected_predictions)

    def test_manifest_records_actual_training_cutoff_parameters_source_and_environment(self):
        manifest = self.manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["format"], "trusted-local-sklearn-pickle")
        self.assertEqual(manifest["feature_schema"], "metro-celsius-calendar-v1")
        self.assertEqual(manifest["model_sha256"], digest(self.original_payload))
        self.assertEqual(manifest["source_sha256"], digest(self.source.read_bytes()))
        self.assertEqual(manifest["training"], {
            "cutoff_exclusive": "2017-01-01",
            "count": 6,
            "first_timestamp": "2016-12-18T08:00:00",
            "last_timestamp": "2016-12-31T23:00:00",
            "validation_used_for_fit": False,
            "reserved_test_used_for_fit": False,
        })
        self.assertEqual(manifest["source_cleaning"], {
            "raw_count": 11,
            "invalid_temp_dropped": 0,
            "duplicate_extras": 1,
            "distinct_timestamps": 10,
        })
        expected_parameters = {
            "n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2,
            "random_state": 42, "n_jobs": 1,
        }
        self.assertEqual(manifest["model_parameters"], expected_parameters)
        actual_parameters = self.fitted_model.named_steps["model"].get_params()
        for name, value in expected_parameters.items():
            self.assertEqual(actual_parameters[name], value)
        self.assertEqual(manifest["prediction_source_sha256"], {
            name: digest((ROOT / "src" / name).read_bytes())
            for name in ("offline_data.py", "data_transformation.py", "evaluation.py")
        })
        self.assertEqual(manifest["environment"], {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "dependencies": {
                name: version(name) for name in (
                    "numpy", "pandas", "scikit-learn", "scipy", "joblib",
                    "cloudpickle", "threadpoolctl", "python-dateutil", "pytz", "six", "tzdata",
                )
            },
        })

    def test_validation_and_reserved_targets_do_not_affect_fitted_predictions(self):
        for year in ("2017", "2018"):
            with self.subTest(year=year):
                rows = [
                    {**row, "traffic_volume": row["traffic_volume"] + 100000}
                    if row["date_time"].startswith(year) else dict(row)
                    for row in self.rows
                ]
                source = self.root / f"changed-{year}.csv"
                write_csv(source, rows)
                directory = self.root / f"changed-{year}"
                pin = artifacts.train_bundle(source, directory)
                model = artifacts.load_bundle(directory, pin)
                self.assertEqual(model.predict(self.features).tolist(), self.expected_predictions)
                manifest = json.loads((directory / "manifest.json").read_bytes())
                self.assertNotEqual(manifest["source_sha256"], self.manifest()["source_sha256"])
                self.assertEqual(manifest["training"], self.manifest()["training"])

    def test_fitted_transformers_use_only_pre_cutoff_rows(self):
        model = artifacts.load_bundle(self.bundle, self.pin)
        transformers = model.named_steps["preprocessor"].named_transformers_
        scaler = transformers["num"].named_steps["scaler"]
        self.assertEqual(scaler.n_samples_seen_, 6)
        self.assertAlmostEqual(scaler.mean_[0], 2.5)
        self.assertEqual(scaler.mean_[1], 25)
        self.assertEqual(transformers["onehot"].named_steps["encoder"].categories_[0].tolist(), ["Clouds"])

    def test_existing_bundle_or_file_is_never_overwritten(self):
        existing_file = self.root / "existing-file"
        existing_file.write_bytes(b"preserve this file")
        empty_directory = self.root / "partial-bundle"
        empty_directory.mkdir()
        for destination in (self.bundle, existing_file, empty_directory):
            with self.subTest(destination=destination.name):
                with patch("src.artifacts.make_model") as factory:
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        artifacts.train_bundle(self.source, destination)
                    factory.assert_not_called()
        self.assertEqual((self.bundle / "manifest.json").read_bytes(), self.original_manifest)
        self.assertEqual((self.bundle / "model.pkl").read_bytes(), self.original_payload)
        self.assertEqual(existing_file.read_bytes(), b"preserve this file")
        self.assertEqual(list(empty_directory.iterdir()), [])

    def test_training_writes_only_requested_bundle_and_does_not_load_legacy_models(self):
        legacy = self.root / "models"
        legacy.mkdir()
        sentinels = {
            "traffic_model.pkl": b"historical model, never deserialize",
            "preprocessor.pkl": b"historical preprocessor, never deserialize",
            "evaluation_results.txt": b"historical result",
        }
        for name, content in sentinels.items():
            (legacy / name).write_bytes(content)
        destination = self.root / "fresh"
        with contextlib.chdir(self.root):
            with patch("src.artifacts.pickle.load") as load, patch("src.artifacts.pickle.loads") as loads:
                artifacts.train_bundle(self.source, destination)
                load.assert_not_called()
                loads.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in legacy.iterdir()}, sentinels)
        self.assertEqual({p.name for p in destination.iterdir()}, {"model.pkl", "manifest.json"})

    def test_missing_pin_does_not_trust_an_adjacent_digest_file(self):
        (self.bundle / "manifest.sha256").write_text(self.pin)
        with patch("src.artifacts.pickle.loads") as loads:
            with self.assertRaises(TypeError):
                artifacts.load_bundle(self.bundle)
            loads.assert_not_called()

    def test_malformed_and_wrong_pins_fail_before_deserialization(self):
        for pin in (None, False, 0, [], b"0" * 64, "", "0" * 63, "0" * 65,
                    "g" * 64, self.pin.upper(), self.pin + "\n", "0" * 64):
            with self.subTest(pin=pin):
                with patch("src.artifacts.pickle.loads") as loads:
                    with self.assertRaises(ValueError):
                        artifacts.load_bundle(self.bundle, pin)
                    loads.assert_not_called()

    def test_tampered_manifest_fails_with_original_independent_pin(self):
        manifest = self.manifest()
        manifest["training"]["count"] += 1
        self.rewrite_manifest(manifest)
        self.assert_rejected_before_unpickling(message="separately trusted pin")

    def test_changed_model_bytes_fail_before_deserialization(self):
        (self.bundle / "model.pkl").write_bytes(self.original_payload + b"changed")
        self.assert_rejected_before_unpickling(message="Model bytes")

    def test_pinned_malformed_json_and_non_object_manifests_fail_before_deserialization(self):
        for encoded in (b"{", b"[]", b"null", b"true", b"1", b'"manifest"'):
            with self.subTest(encoded=encoded):
                (self.bundle / "manifest.json").write_bytes(encoded)
                self.assert_rejected_before_unpickling(digest(encoded))

    def test_unsupported_versions_formats_and_feature_schemas_fail_before_deserialization(self):
        for field, values in (
            ("schema_version", (None, True, "1", 1.0, 0, 2)),
            ("format", (None, "pickle", "untrusted-pickle")),
            ("feature_schema", (None, "metro-kelvin-calendar-v1")),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    manifest = self.manifest()
                    manifest[field] = value
                    self.assert_rejected_before_unpickling(self.rewrite_manifest(manifest))

    def test_python_and_dependency_mismatches_fail_before_deserialization(self):
        for mismatch in ("python", "version", "missing-dependency", "missing-environment"):
            with self.subTest(mismatch=mismatch):
                manifest = self.manifest()
                if mismatch == "python":
                    manifest["environment"]["python_major_minor"] = "0.0"
                elif mismatch == "version":
                    manifest["environment"]["dependencies"]["scikit-learn"] = "0.0"
                elif mismatch == "missing-dependency":
                    del manifest["environment"]["dependencies"]["numpy"]
                else:
                    del manifest["environment"]
                self.assert_rejected_before_unpickling(self.rewrite_manifest(manifest), message="runtime differs")

    def test_source_mismatches_fail_before_deserialization(self):
        for mismatch in ("changed-digest", "missing-source", "missing-metadata"):
            with self.subTest(mismatch=mismatch):
                manifest = self.manifest()
                if mismatch == "changed-digest":
                    manifest["prediction_source_sha256"]["offline_data.py"] = "0" * 64
                elif mismatch == "missing-source":
                    del manifest["prediction_source_sha256"]["data_transformation.py"]
                else:
                    del manifest["prediction_source_sha256"]
                self.assert_rejected_before_unpickling(self.rewrite_manifest(manifest), message="Prediction source")

    def test_invalid_or_missing_payload_digest_fails_before_deserialization(self):
        for value in (None, "", "g" * 64, "0" * 64, 0, []):
            with self.subTest(value=value):
                manifest = self.manifest()
                manifest["model_sha256"] = value
                self.assert_rejected_before_unpickling(self.rewrite_manifest(manifest), message="Model bytes")

    def test_size_limits_reject_before_deserialization_and_allow_exact_boundaries(self):
        for constant, data in (
            ("MAX_MANIFEST_BYTES", self.original_manifest),
            ("MAX_MODEL_BYTES", self.original_payload),
        ):
            with self.subTest(constant=constant):
                with patch.object(artifacts, constant, len(data) - 1):
                    self.assert_rejected_before_unpickling(message="size limit")
                with patch.object(artifacts, constant, len(data)):
                    model = artifacts.load_bundle(self.bundle, self.pin)
                    self.assertEqual(model.predict(self.features).tolist(), self.expected_predictions)

    def test_training_rejects_oversized_payload_without_creating_bundle(self):
        destination = self.root / "too-large"
        with patch.object(artifacts, "MAX_MODEL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "Model exceeds"):
                artifacts.train_bundle(self.source, destination)
        self.assertFalse(destination.exists())

    def test_missing_bundle_and_partial_directories_fail_before_deserialization(self):
        self.assert_rejected_before_unpickling(directory=self.root / "missing")
        empty = self.root / "empty"
        empty.mkdir()
        self.assert_rejected_before_unpickling(directory=empty)
        for missing in ("manifest.json", "model.pkl"):
            with self.subTest(missing=missing):
                partial = self.root / f"without-{missing}"
                shutil.copytree(self.fixture_bundle, partial)
                (partial / missing).unlink()
                self.assert_rejected_before_unpickling(directory=partial)

    def test_bundle_files_cannot_be_directories_or_symlinks(self):
        for name in ("manifest.json", "model.pkl"):
            for kind in ("directory", "symlink", "dangling-symlink"):
                with self.subTest(name=name, kind=kind):
                    directory = self.root / f"{kind}-{name}"
                    shutil.copytree(self.fixture_bundle, directory)
                    target = directory / name
                    target.unlink()
                    if kind == "directory":
                        target.mkdir()
                    elif kind == "symlink":
                        target.symlink_to(self.fixture_bundle / name)
                    else:
                        target.symlink_to(self.root / "nonexistent")
                    self.assert_rejected_before_unpickling(directory=directory, message="non-symlink")

    def test_manifest_cannot_redirect_model_loading_outside_fixed_filename(self):
        outside = self.root / "outside.pkl"
        outside.write_bytes(b"must not be deserialized")
        manifest = self.manifest()
        manifest.update({"model_file": "../outside.pkl", "model_path": str(outside)})
        pin = self.rewrite_manifest(manifest)
        with patch("src.artifacts.pickle.loads", wraps=pickle.loads) as loads:
            restored = artifacts.load_bundle(self.bundle, pin)
        loads.assert_called_once_with(self.original_payload)
        self.assertEqual(restored.predict(self.features).tolist(), self.expected_predictions)
        (self.bundle / "model.pkl").unlink()
        manifest["model_sha256"] = digest(outside.read_bytes())
        self.assert_rejected_before_unpickling(self.rewrite_manifest(manifest), message="model.pkl")
        self.assertEqual(outside.read_bytes(), b"must not be deserialized")

    def test_only_trusted_fresh_expected_pipeline_is_accepted_after_loading(self):
        for value in ({"not": "a pipeline"}, Pipeline([])):
            with self.subTest(value=type(value).__name__):
                payload = pickle.dumps(value, protocol=5)
                (self.bundle / "model.pkl").write_bytes(payload)
                manifest = self.manifest()
                manifest["model_sha256"] = digest(payload)
                pin = self.rewrite_manifest(manifest)
                with patch("src.artifacts.pickle.loads", wraps=pickle.loads) as loads:
                    with self.assertRaisesRegex(ValueError, "expected prediction pipeline"):
                        artifacts.load_bundle(self.bundle, pin)
                loads.assert_called_once_with(payload)


if __name__ == "__main__":
    unittest.main()
