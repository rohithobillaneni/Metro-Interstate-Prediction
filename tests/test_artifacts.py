import contextlib
import copy
import csv
import hashlib
import io
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

    def test_training_rejects_provenance_changes_during_fit_or_serialization(self):
        for stage in ("fit", "serialization"):
            for field in ("source", "environment", "parameters"):
                with self.subTest(stage=stage, field=field), contextlib.ExitStack() as stack:
                    destination = self.root / f"drift-{stage}-{field}"
                    source_hashes = artifacts._source_hashes()
                    environment = artifacts._environment()
                    stack.enter_context(patch(
                        "src.artifacts._source_hashes", side_effect=lambda: dict(source_hashes)
                    ))
                    stack.enter_context(patch(
                        "src.artifacts._environment", side_effect=lambda: copy.deepcopy(environment)
                    ))
                    stack.enter_context(patch.dict(artifacts.MODEL_PARAMETERS))

                    def mutate():
                        if field == "source":
                            source_hashes["offline_data.py"] = "0" * 64
                        elif field == "environment":
                            environment["dependencies"]["numpy"] = "changed-during-training"
                        else:
                            artifacts.MODEL_PARAMETERS["max_depth"] += 1

                    model = make_model()
                    original = Pipeline.fit if stage == "fit" else pickle.dumps

                    def change_after_work(*args, **kwargs):
                        result = original(*args, **kwargs)
                        mutate()
                        return result

                    stack.enter_context(patch("src.artifacts.make_model", return_value=model))
                    if stage == "fit":
                        stack.enter_context(patch.object(Pipeline, "fit", new=change_after_work))
                    else:
                        stack.enter_context(patch("src.artifacts.pickle.dumps", side_effect=change_after_work))
                    with self.assertRaisesRegex(ValueError, "changed while training"):
                        artifacts.train_bundle(self.source, destination)
                    self.assertFalse(destination.exists())

    def test_training_rejects_csv_changes_during_loading_or_fitting(self):
        for stage in ("loading", "fitting"):
            with self.subTest(stage=stage), contextlib.ExitStack() as stack:
                source = self.root / f"drifting-{stage}.csv"
                write_csv(source, self.rows)
                destination = self.root / f"drifting-{stage}"
                model = make_model()
                original = load_csv if stage == "loading" else Pipeline.fit

                def change_after_read(*args, **kwargs):
                    result = original(*args, **kwargs)
                    source.write_bytes(source.read_bytes() + b"\n")
                    return result

                stack.enter_context(patch("src.artifacts.make_model", return_value=model))
                if stage == "loading":
                    stack.enter_context(patch("src.artifacts.load_csv", side_effect=change_after_read))
                else:
                    stack.enter_context(patch.object(Pipeline, "fit", new=change_after_read))
                with self.assertRaisesRegex(ValueError, "Source CSV changed while"):
                    artifacts.train_bundle(source, destination)
                self.assertFalse(destination.exists())

    def test_fit_and_serialization_failures_do_not_create_a_bundle(self):
        for stage in ("fit", "serialization"):
            with self.subTest(stage=stage), contextlib.ExitStack() as stack:
                destination = self.root / f"failed-{stage}"
                model = make_model()
                failure = ValueError(f"injected {stage} failure")
                stack.enter_context(patch("src.artifacts.make_model", return_value=model))
                if stage == "fit":
                    stack.enter_context(patch.object(model, "fit", side_effect=failure))
                else:
                    stack.enter_context(patch("src.artifacts.pickle.dumps", side_effect=failure))
                with self.assertRaises(ValueError) as raised:
                    artifacts.train_bundle(self.source, destination)
                self.assertIs(raised.exception, failure)
                self.assertFalse(destination.exists())

    def test_destination_created_during_fit_is_not_overwritten(self):
        destination = self.root / "racing-destination"
        model = make_model()
        original_fit = Pipeline.fit

        def create_destination(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            destination.mkdir()
            (destination / "sentinel").write_bytes(b"another writer")
            return result

        with patch("src.artifacts.make_model", return_value=model):
            with patch.object(Pipeline, "fit", new=create_destination):
                with self.assertRaises(FileExistsError):
                    artifacts.train_bundle(self.source, destination)
        self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()},
                         {"sentinel": b"another writer"})

    def test_cli_manifest_write_failure_returns_no_pin_and_partial_bundle_fails_closed(self):
        destination = self.root / "failed-write"
        original_open = Path.open

        def fail_manifest_open(path, *args, **kwargs):
            if path == destination / "manifest.json":
                raise OSError("injected disk failure")
            return original_open(path, *args, **kwargs)

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(Path, "open", fail_manifest_open):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    artifacts.main(["--data", str(self.source), "--output", str(destination)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("injected disk failure", stderr.getvalue())
        self.assertEqual({p.name for p in destination.iterdir()}, {"model.pkl"})
        self.assert_rejected_before_unpickling(directory=destination, message="manifest.json")
        self.assertEqual((self.bundle / "manifest.json").read_bytes(), self.original_manifest)
        self.assertEqual((self.bundle / "model.pkl").read_bytes(), self.original_payload)

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
