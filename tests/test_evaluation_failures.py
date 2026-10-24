"""Report-boundary checks with synthetic data and cached real fold results."""
import contextlib
import copy
import csv
from datetime import datetime, timedelta
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import temporal_evaluation as temporal, weather_sensitivity as weather


ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = (temporal, weather)


class EvaluationFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture_directory.cleanup)
        cls.source = Path(cls.fixture_directory.name) / "data.csv"
        stamps = {"2016-12-25 00:00:00", "2016-12-31 23:00:00"}
        for start, _ in (*temporal.DEVELOPMENT_WINDOWS, *weather.WINDOWS):
            origin = datetime.fromisoformat(start)
            stamps.update((origin.strftime("%Y-%m-%d %H:%M:%S"),
                           (origin + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")))
        with cls.source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("date_time", "traffic_volume", "temp", "clouds_all", "weather_main", "holiday"))
            for index, stamp in enumerate(sorted(stamps)):
                writer.writerow((stamp, index * 10, 290, 30, "Clouds", "None"))
                writer.writerow((stamp, index * 10, 280, 30, "Rain", "None"))
            writer.writerow(("2018-01-01 00:00:00", "unscored", "invalid", "invalid", "", ""))
        cls.fold_results = {}
        # Fit once per real fold, then inject faults after this boundary. Existing
        # numerical tests and full-data CI still exercise fitting on every run.
        for module in EVALUATORS:
            results = []
            original = module.evaluate_folds

            def capture(folds):
                result = original(folds)
                results.append(result)
                return result

            with patch.object(module, "evaluate_folds", side_effect=capture):
                module.evaluate(cls.source)
            cls.fold_results[module] = results

    @contextlib.contextmanager
    def workspace(self, module, mutate=None):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            (root / "src").mkdir()
            for path in temporal._source_hashes():
                (root / path).write_bytes((ROOT / path).read_bytes())
            lock = root / "requirements-evaluation.txt"
            lock.write_bytes((ROOT / lock.name).read_bytes())
            protocol = root / "protocol.json"
            protocol.write_bytes(weather.PROTOCOL_PATH.read_bytes())
            source = root / "data.csv"
            source.write_bytes(self.source.read_bytes())
            stack.enter_context(patch.object(temporal, "ROOT", root))
            stack.enter_context(patch.object(weather, "ROOT", root))
            stack.enter_context(patch.object(weather, "PROTOCOL_PATH", protocol))
            stack.enter_context(patch.dict(temporal.MODEL_PARAMETERS))
            calls = iter(copy.deepcopy(self.fold_results[module]))
            changed = False

            def at_fold_boundary(folds):
                nonlocal changed
                result = next(calls)
                if mutate is not None and not changed:
                    mutate(root)
                    changed = True
                return result

            stack.enter_context(patch.object(module, "evaluate_folds", side_effect=at_fold_boundary))
            yield root, source

    def test_lock_versions_and_parameters_cannot_change_during_evaluation(self):
        for module in EVALUATORS:
            for field in ("lock", "versions", "parameters"):
                with self.subTest(evaluator=module.__name__, field=field):
                    changed = False
                    original_version = temporal.version

                    def version(name):
                        return "changed-during-evaluation" if changed and name == "numpy" else original_version(name)

                    def mutate(root):
                        nonlocal changed
                        changed = True
                        if field == "lock":
                            with (root / "requirements-evaluation.txt").open("a") as stream:
                                stream.write("\n# changed during evaluation\n")
                        elif field == "parameters":
                            temporal.MODEL_PARAMETERS["max_depth"] += 1

                    with contextlib.ExitStack() as stack:
                        if field == "versions":
                            stack.enter_context(patch.object(temporal, "version", side_effect=version))
                            stack.enter_context(patch.object(weather, "version", side_effect=version, create=True))
                        with self.workspace(module, mutate) as (_, source):
                            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                                module.evaluate(source)

    def test_csv_implementation_and_protocol_drift_return_no_cli_report(self):
        for module in EVALUATORS:
            fields = ("csv", "implementation", "protocol") if module is weather else ("csv", "implementation")
            for field in fields:
                with self.subTest(evaluator=module.__name__, field=field):
                    def mutate(root):
                        path = root / {"csv": "data.csv", "implementation": "src/temporal_evaluation.py",
                                       "protocol": "protocol.json"}[field]
                        path.write_bytes(path.read_bytes() + b"\n")

                    with self.workspace(module, mutate) as (root, source):
                        output = root / "report.json"
                        stdout, stderr = io.StringIO(), io.StringIO()
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            with self.assertRaises(SystemExit) as raised:
                                module.main(["--data", str(source), "--output", str(output)])
                        self.assertEqual(raised.exception.code, 2)
                        self.assertIn("changed during evaluation", stderr.getvalue())
                        self.assertEqual(stdout.getvalue(), "")
                        self.assertFalse(output.exists())

    def test_csv_change_during_loading_fails_before_fold_computation(self):
        for module in EVALUATORS:
            with self.subTest(evaluator=module.__name__), self.workspace(module) as (_, source):
                original = module.load_development_csv

                def mutate_after_loading(*args, **kwargs):
                    result = original(*args, **kwargs)
                    source.write_bytes(source.read_bytes() + b"\n")
                    return result

                with patch.object(module, "load_development_csv", side_effect=mutate_after_loading):
                    with self.assertRaisesRegex(ValueError, "changed while loading"):
                        module.evaluate(source)
                module.evaluate_folds.assert_not_called()

    def test_computation_and_nonfinite_serialization_fail_without_output(self):
        for module in EVALUATORS:
            for failure in ("computation", "nonfinite-report"):
                with self.subTest(evaluator=module.__name__, failure=failure):
                    with self.workspace(module) as (root, source), contextlib.ExitStack() as stack:
                        if failure == "computation":
                            stack.enter_context(patch.object(module, "evaluate_folds", side_effect=ValueError("injected fold failure")))
                        else:
                            stack.enter_context(patch.object(module, "evaluate", return_value={"mae": float("nan")}))
                        output = root / "report.json"
                        stdout, stderr = io.StringIO(), io.StringIO()
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            with self.assertRaises(SystemExit) as raised:
                                module.main(["--data", str(source), "--output", str(output)])
                        self.assertEqual(raised.exception.code, 2)
                        self.assertEqual(stdout.getvalue(), "")
                        self.assertFalse(output.exists())

    def test_destination_created_during_computation_is_not_overwritten(self):
        for module in EVALUATORS:
            with self.subTest(evaluator=module.__name__):
                def create_output(root):
                    (root / "report.json").write_bytes(b"another writer's evidence")

                with self.workspace(module, create_output) as (root, source):
                    output = root / "report.json"
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            module.main(["--data", str(source), "--output", str(output)])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertEqual(output.read_bytes(), b"another writer's evidence")


if __name__ == "__main__":
    unittest.main()
