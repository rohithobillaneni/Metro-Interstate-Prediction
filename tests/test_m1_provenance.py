"""M1 report provenance checks using real fitting on small synthetic data."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import evaluation


class M1ProvenanceTests(unittest.TestCase):
    @contextlib.contextmanager
    def workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("date_time", "traffic_volume", "temp", "clouds_all", "weather_main", "holiday"))
                for year, day, target in ((2016, 25, 10), (2016, 26, 30),
                                          (2016, 27, 50), (2017, 1, 20),
                                          (2017, 2, 40), (2018, 1, 999)):
                    writer.writerow((f"{year}-01-{day:02d} 08:00:00", target, 290, 30, "Clouds", "None"))
            yield root, source

    @contextlib.contextmanager
    def mutate_after(self, boundary, mutate):
        original = getattr(evaluation.Pipeline, boundary)

        def wrapped(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            mutate()
            return result

        with patch.object(evaluation.Pipeline, boundary, new=wrapped):
            yield

    def test_parameter_and_version_drift_is_rejected_after_fit_and_prediction(self):
        for boundary in ("fit", "predict"):
            for field in ("parameters", "versions"):
                with self.subTest(boundary=boundary, field=field), self.workspace() as (_, source):
                    changed = False
                    original_version = evaluation.version

                    def version(name):
                        return "changed-during-evaluation" if changed and name == "numpy" else original_version(name)

                    def mutate():
                        nonlocal changed
                        changed = True
                        if field == "parameters":
                            evaluation.MODEL_PARAMETERS["max_depth"] += 1

                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.dict(evaluation.MODEL_PARAMETERS))
                        if field == "versions":
                            stack.enter_context(patch.object(evaluation, "version", side_effect=version))
                        stack.enter_context(self.mutate_after(boundary, mutate))
                        with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                            evaluation.evaluate(source)

    def test_csv_drift_at_computation_boundaries_emits_no_report(self):
        for boundary in ("fit", "predict"):
            for destination in ("file", "stdout"):
                with self.subTest(boundary=boundary, destination=destination), self.workspace() as (root, source):
                    def mutate():
                        source.write_bytes(source.read_bytes() + b"\n")

                    output = root / "report.json"
                    args = ["--data", str(source)]
                    if destination == "file":
                        args += ["--output", str(output)]
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with self.mutate_after(boundary, mutate), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            evaluation.main(args)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("changed during evaluation", stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertFalse(output.exists())

    def test_csv_change_during_loading_fails_before_model_creation(self):
        with self.workspace() as (_, source):
            original = evaluation.load_csv

            def load_then_mutate(path):
                result = original(path)
                source.write_bytes(source.read_bytes() + b"\n")
                return result

            with patch.object(evaluation, "load_csv", side_effect=load_then_mutate):
                with patch.object(evaluation, "make_model") as make_model:
                    with self.assertRaisesRegex(ValueError, "changed while loading"):
                        evaluation.evaluate(source)
            make_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
