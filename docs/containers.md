# Reproduce the local container stack

This is a local traffic-volume reference application. It is not authenticated
or suitable for exposure to the public internet. Docker Engine and Compose v2
are required. Images are built locally; no registry push, cloud service, paid
API or prebuilt model download is required.

## One-command integration check

From the repository root, on a host with Python3 and a running Docker engine:

```sh
python3 scripts/container_smoke.py
```

This builds both images, trains a fresh model using the backend image and public
CSV, starts an isolated Compose project on ephemeral loopback ports, checks API
validation and prediction, and executes a Streamlit session against the real
API. It also checks rejection of a wrong manifest pin. The output records the
prediction and manifest digest, not a benchmark accuracy claim.

The script collects logs, then removes only its uniquely named containers,
network, temporary bundle and image tags. It never prunes Docker resources or
uploads models, images or artifacts. Base/build-cache layers may remain.
The `Container reproduction` CI workflow runs this same check on a standard
Linux runner; other platforms require their own runtime verification.

## Keep a local stack running

Choose an unused Compose project name and a new model output directory:

```sh
export COMPOSE_PROJECT_NAME=metro-local
# Build does not need a model; API startup requires an explicit trusted pin.
docker compose build
mkdir -p generated-models
docker run --rm --network none --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=${PWD}/data/Metro_Interstate_Traffic_Volume.csv,target=/input.csv,readonly" \
  --mount "type=bind,source=${PWD}/generated-models,target=/output" \
  "${COMPOSE_PROJECT_NAME}-backend:local" \
  python -m src.artifacts --data /input.csv --output /output/container-v1
```

The training command prints `manifest_sha256`. It fits only pre-2017 rows and
refuses to replace an existing output directory. Copy the digest from **this
reviewed local training run**, then start the same backend image:

```sh
export METRO_MODEL_HOST_DIR="${PWD}/generated-models/container-v1"
export METRO_MODEL_MANIFEST_SHA256=PASTE_THE_DIGEST_FROM_YOUR_LOCAL_TRAINING_RUN
docker compose up --detach --wait --wait-timeout 120
```

Open `http://127.0.0.1:8501`; the API is at `http://127.0.0.1:8000`.
Set `METRO_FRONTEND_PORT` or `METRO_BACKEND_PORT` before starting if those host
ports are occupied. Frontend-to-backend requests use `http://backend:8000/predict`
inside the project network. The frontend waits for backend health; startup
ordering alone would not verify readiness.

Stop this project without deleting the generated model:

```sh
docker compose down
```

## Trust and reproducibility boundaries

The backend verifies the separately configured manifest digest, model bytes,
Python major/minor, exact evaluation dependencies and source hashes. Train and
serve with the same image: an old host model or a bundle from changed source
may correctly fail startup. A checksum does not make an untrusted pickle safe;
read [the artifact trust boundary](serving.md). Missing files, missing pin or a
wrong pin fail closed. Compose will not silently create a missing model directory.

Both services run as UID/GID10001, with read-only root filesystems, temporary
`/tmp`, dropped capabilities and no new privileges. The model bind is read-only.
The training command uses the host user's UID/GID to avoid root-owned output;
keep bundle directories/files readable by the runtime user. No historical
pickle, dataset, environment, credential or local model is copied into images.
The frontend includes its shared feature module and disables usage telemetry;
CORS/XSRF protections remain enabled.

The Dockerfiles pin the official Python3.13-slim multi-platform manifest digest.
Dependency hashes pin packages separately. The initial digest was resolved from
the official `library/python` registry manifest and independently hashed on
23 October2026. Pins require reviewed security updates; they do not establish
vulnerability absence or identical floating-point predictions across machines.
Dependency licence notices are retained in their installed distributions.

The historical Cassandra ingestion/training paths and saved models remain
outside this reproduction contract. No claim is made about deployment capacity,
forecasting accuracy or browser visual coverage.
