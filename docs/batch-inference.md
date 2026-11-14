# Predict a batch without running the API

This command is available on **main (unreleased)**, not in v0.1.2. It uses the
existing trained bundle and API request schema to estimate multiple independent
same-hour traffic volumes locally. It does not train, score targets, call a
service or establish advance forecasting accuracy.

Complete the [maintained setup](../README.md#maintained-setup), then switch this
checkout to `main` for the new command. The same locked serving dependencies
apply; no additional package is needed. Train a fresh bundle using the
[local training instructions](serving.md#train-a-fresh-bundle) and retain the
returned manifest digest separately.

**Only use your own reviewed training bundle. Pickle can execute code; a digest
does not make an untrusted model safe.** The command uses the same pin, runtime,
source and model-byte verification as API startup. It never trusts a digest
discovered beside a downloaded artifact or falls back to historical pickles.

## Input and command

Save a UTF-8 JSON array as `requests.json`. Each object follows the existing
[`/predict` request contract](serving.md#start-and-query). For example, these are
synthetic requests, not an evaluation cohort:

```json
[
  {"temp":15,"clouds_all":50,"holiday":"No","weather_main":"Clouds","weekday":"Monday","hour":12,"month":"January"},
  {"temp":-5,"clouds_all":90,"holiday":"No","weather_main":"Snow","weekday":"Tuesday","hour":23,"month":"January"}
]
```

From the repository root, with `METRO_MODEL_DIR` and
`METRO_MODEL_MANIFEST_SHA256` set from your local training run:

```sh
.venv/bin/python -m src.predict \
  --input requests.json \
  --model-dir "$METRO_MODEL_DIR" \
  --manifest-sha256 "$METRO_MODEL_MANIFEST_SHA256" \
  --output predictions.json
```

Omit `--output` to print the JSON report to standard output. The command does not
start an HTTP server. Temperature is Celsius. Numeric strings and booleans are
invalid, `hour_category` is optional but must agree with `hour`, and unknown
weather labels retain the fitted encoder's existing handling. Target columns,
such as `traffic_volume`, are rejected as unknown fields. Raw UCI CSV files are
not batch inputs; all benchmark protocols and the unscored 2018 boundary remain
unchanged.

## Output and failure behavior

The report has `schema_version: 1`, the same-hour estimation task description,
`input_sha256`, `manifest_sha256`, `count` and a `predictions` array. Each entry
contains `row` (the **one-based input array position**) and `prediction` in
vehicles/hour. Input order and duplicates are retained; no rows are silently
dropped. Predictions match the API when it uses the same bundle and requests.
No target metrics are computed, and raw request values are not copied into the
report.

The input digest identifies the exact bytes read and parsed, including whitespace.
If the file changes afterward, it does not change that in-memory batch. Keep the
original input, separately retained manifest pin, bundle and checkout revision
with the report. These digests identify inputs; they do not authenticate an
author, attest loaded code or prove prediction quality.

Input is limited to **8 MiB and 10,000 requests** and must contain at least one
request. The command validates the whole batch before loading the model or
predicting. Malformed UTF-8/JSON, duplicate object keys, invalid requests and
wrong-shaped, nonfinite or negative model outputs fail with a nonzero exit and
no result report. Request-schema errors identify the row without echoing its
values. A model is loaded once and predicts the batch in one call; these bounds
are not a throughput claim.

Existing output files are never overwritten, including if another writer creates
one during prediction. A complete result is serialized before a new output is
opened exclusively. A failed disk write can leave an incomplete new file;
preserve it for diagnosis and choose another path. This is a small local batch
interface, not streaming inference or an authenticated service.
