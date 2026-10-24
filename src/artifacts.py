"""Generate and load explicitly trusted local model bundles.

Pickle is executable, not a safe interchange format. The separately supplied
manifest digest must come from your own reviewed training run, never from an
untrusted artifact supplier or an automatically trusted adjacent file.
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import re

import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation import MODEL_PARAMETERS, make_model
from src.offline_data import load_csv, partition_records


DEPENDENCIES = (
    "numpy", "pandas", "scikit-learn", "scipy", "joblib", "cloudpickle",
    "threadpoolctl", "python-dateutil", "pytz", "six", "tzdata",
)
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MODEL_BYTES = 128 * 1024 * 1024


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: _digest((root / name).read_bytes())
        for name in ("offline_data.py", "data_transformation.py", "evaluation.py")
    }


def _environment() -> dict:
    return {
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "dependencies": {name: version(name) for name in DEPENDENCIES},
    }


def train_bundle(source: str | Path, directory: str | Path) -> str:
    """Fit pre-2017 only and return the manifest pin for a new local directory."""
    source, directory = Path(source), Path(directory)
    if directory.exists():
        raise ValueError("Bundle directory already exists; previous artifacts are never replaced")
    source_digest = _digest(source.read_bytes())
    source_hashes = _source_hashes()
    environment = _environment()
    model_parameters = MODEL_PARAMETERS.copy()
    records, diagnostics = load_csv(source)
    if _digest(source.read_bytes()) != source_digest:
        raise ValueError("Source CSV changed while loading")
    parts = partition_records(records)
    model = make_model()
    model.fit(
        pd.DataFrame([record.features() for record in parts.train]),
        [record.traffic_volume for record in parts.train],
    )
    payload = pickle.dumps(model, protocol=5)
    if len(payload) > MAX_MODEL_BYTES:
        raise ValueError("Model exceeds the supported bundle size")
    # Bind the manifest to the pre-training snapshot, not metadata first read
    # after fitting. These drift checks are not loaded-code attestation.
    if _digest(source.read_bytes()) != source_digest:
        raise ValueError("Source CSV changed while training")
    if _source_hashes() != source_hashes:
        raise ValueError("Prediction source changed while training")
    if _environment() != environment:
        raise ValueError("Python/dependency metadata changed while training")
    if MODEL_PARAMETERS != model_parameters:
        raise ValueError("Model parameters changed while training")
    manifest = {
        "schema_version": 1,
        "format": "trusted-local-sklearn-pickle",
        "feature_schema": "metro-celsius-calendar-v1",
        "model_sha256": _digest(payload),
        "source_sha256": source_digest,
        "training": {
            "cutoff_exclusive": "2017-01-01",
            "count": len(parts.train),
            "first_timestamp": parts.train[0].date_time.isoformat(),
            "last_timestamp": parts.train[-1].date_time.isoformat(),
            "validation_used_for_fit": False,
            "reserved_test_used_for_fit": False,
        },
        "source_cleaning": asdict(diagnostics),
        "model_parameters": model_parameters,
        "prediction_source_sha256": source_hashes,
        "environment": environment,
    }
    encoded = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # Exclusive directory creation prevents replacement, including by racing runs.
    # A failed write can leave an incomplete directory; loading it must fail closed.
    directory.mkdir()
    with (directory / "model.pkl").open("xb") as output:
        output.write(payload)
    with (directory / "manifest.json").open("xb") as output:
        output.write(encoded)
    return _digest(encoded)


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Bundle requires a regular non-symlink {path.name}")
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Bundle {path.name} exceeds its size limit")
    return data


def load_bundle(directory: str | Path, trusted_manifest_sha256: str) -> Pipeline:
    """Verify an independently trusted pin before deserializing exact bytes."""
    if not isinstance(trusted_manifest_sha256, str) or not re.fullmatch(
        "[0-9a-f]{64}", trusted_manifest_sha256
    ):
        raise ValueError("A separately trusted lowercase SHA256 manifest pin is required")
    directory = Path(directory)
    encoded = _read_bounded(directory / "manifest.json", MAX_MANIFEST_BYTES)
    if _digest(encoded) != trusted_manifest_sha256:
        raise ValueError("Manifest does not match the separately trusted pin")
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("Invalid model manifest")
    if manifest.get("schema_version") != 1 or manifest.get("format") != "trusted-local-sklearn-pickle":
        raise ValueError("Unsupported model bundle version or format")
    if manifest.get("feature_schema") != "metro-celsius-calendar-v1":
        raise ValueError("Unsupported model feature schema")
    if manifest.get("environment") != _environment():
        raise ValueError("Model runtime differs from its recorded Python/dependency versions")
    if manifest.get("prediction_source_sha256") != _source_hashes():
        raise ValueError("Prediction source differs from the training bundle")
    payload = _read_bounded(directory / "model.pkl", MAX_MODEL_BYTES)
    if _digest(payload) != manifest.get("model_sha256"):
        raise ValueError("Model bytes do not match the pinned manifest")
    # Intentional trusted-local deserialization. Hashing and loading use the same
    # byte buffer; no pathname is reopened after integrity verification.
    model = pickle.loads(payload)
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["preprocessor", "model"]:
        raise ValueError("Bundle does not contain the expected prediction pipeline")
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local bundle directory")
    args = parser.parse_args(argv)
    try:
        pin = train_bundle(args.data, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest_sha256": pin, "training_cutoff_exclusive": "2017-01-01"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
