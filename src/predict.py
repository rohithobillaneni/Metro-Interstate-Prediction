"""Predict an ordered batch locally from an explicitly trusted model bundle."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.app import TrafficData
from src.artifacts import load_bundle


MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_REQUESTS = 10_000


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Input JSON must not contain duplicate object keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Input JSON must not contain NaN or Infinity")


def predict_file(
    source: str | Path, model_directory: str | Path, trusted_manifest_sha256: str,
) -> dict:
    """Validate all requests, then predict once without starting an HTTP server.

    Read and hash the same bounded input bytes. Row numbers are one-based array
    positions; requests are never sorted, deduplicated or silently dropped.
    """
    with Path(source).open("rb") as stream:
        encoded = stream.read(MAX_INPUT_BYTES + 1)
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError(f"Input exceeds the {MAX_INPUT_BYTES}-byte limit")
    try:
        requests = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RecursionError as exc:
        raise ValueError("Input JSON nesting exceeds the supported limit") from exc
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_REQUESTS:
        raise ValueError(f"Input must be a JSON array with 1 to {MAX_REQUESTS} requests")
    features = []
    for row, request in enumerate(requests, start=1):
        try:
            features.append(TrafficData.model_validate(request).features())
        except ValidationError as exc:
            # Identify the offending row without echoing arbitrary input values.
            raise ValueError(f"Request {row} does not match the prediction API schema") from exc
    model = load_bundle(model_directory, trusted_manifest_sha256)
    values = np.asarray(model.predict(pd.DataFrame(features)))
    if values.ndim != 1 or len(values) != len(features):
        raise ValueError("Model must return one prediction per request")
    if values.dtype.kind not in "iuf":
        raise ValueError("Model returned invalid traffic volume")
    predictions = []
    for row, value in enumerate(values, start=1):
        prediction = float(value)
        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError("Model returned invalid traffic volume")
        predictions.append({"row": row, "prediction": prediction})
    return {
        "schema_version": 1,
        "task": "same-hour traffic estimation with observed weather; not advance forecasting",
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest_sha256": trusted_manifest_sha256,
        "count": len(predictions),
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 JSON array of prediction API requests")
    parser.add_argument("--model-dir", type=Path, required=True, help="Your reviewed local training bundle")
    parser.add_argument("--manifest-sha256", required=True, help="Digest retained separately from your training run")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new path to preserve previous results")
    try:
        report = predict_file(args.input, args.model_dir, args.manifest_sha256)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
