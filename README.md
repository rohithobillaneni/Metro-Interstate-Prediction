# Metro Traffic Volume Prediction

A Python reference project for estimating hourly I-94 traffic volume from
calendar and observed weather features, using the public UCI dataset.
The maintained scope is offline evaluation and trusted local API/frontend
reproduction. This is retrospective same-hour estimation, not an advance
forecast or an authenticated public service.

## Maintained setup

Use a source checkout with a security-maintained Python 3.13 patch that meets
the [interpreter requirement in SECURITY.md](SECURITY.md). Check the actual
interpreter before creating the environment and check an existing environment
again after updating Python. Dependency installation does not patch Python.
The commands below use a POSIX shell and explicitly select the local environment;
run subsequent commands from the repository root.

```sh
git clone https://github.com/rohithobillaneni/Metro-Interstate-Prediction.git
cd Metro-Interstate-Prediction
python3.13 --version
python3.13 -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-serving.txt -r requirements-frontend.txt
.venv/bin/python -m unittest discover -s tests -v
```

Installation needs package access. Evaluation and local model training need no
Cassandra instance, API credentials or external service. These hash-locked
dependencies cover evaluation, API and frontend tests. The historical
`requirements.txt` and `setup.py` are not this installation contract; a package
registry installation is not currently supported. See the
[architecture and supported interfaces](docs/architecture.md).

## Evaluate

```sh
.venv/bin/python -m src.evaluation --data data/Metro_Interstate_Traffic_Volume.csv --output validation-result.json
```

This fits on pre-2017 data and compares a seeded forest with training-only mean
and weekday/hour baselines on 2017. **2018 remains unscored.** The bundled raw
CSV is the input; historical processed/test CSVs and saved models are not used.
Choose a separate output filename for each report you want to retain.

| Evaluation | Purpose and reproduction |
|---|---|
| Fixed-year comparison | [M1 protocol and limitations](docs/evaluation.md); the command above |
| Temporal development validation | [Four expanding-window weeks and exact seasonal baseline](docs/temporal-evaluation.md) |
| Weather-row sensitivity | [53 fixed folds covering retained 2017 hours, paired first/last policies](docs/weather-sensitivity.md) |

The committed reports retain unfavorable comparisons: the seasonal baseline has
lower pooled RMSE in the four-week experiment, and the first-row forest has
higher Q4 MAE than the calendar baseline in the full-year experiment. These are
different cohorts. 2017 is already used development data, not an untouched final
test. Missing hours, naive local-clock timestamps and duplicate weather-row
choices limit interpretation; the linked protocols describe these boundaries.

The [technical report](docs/technical-report.md) brings these recorded results
together, explains the comparison boundaries and preserves the negative findings.

## Run locally

After the maintained setup, follow [local inference and API startup](docs/serving.md)
to train a fresh bundle, retain its manifest digest separately, start the API on
loopback, and run the frontend. A bundle's output directory must be new. Only
load your own reviewed training output: pickle can execute code, and a checksum
does not make an untrusted model safe. Historical pickles are never a fallback.

For an isolated Linux runtime, use [container reproduction](docs/containers.md).
That guide requires Docker Engine and Compose v2 and trains the model inside the
same backend image used for serving. It is a local reproduction path, not a
public deployment guide.

## Data, contributions and history

Source code is [MIT licensed](LICENSE). The bundled UCI Metro Interstate Traffic
Volume dataset is separately licensed under **CC BY 4.0**; see
[DATASET.md](DATASET.md) for provenance, attribution and redistribution terms.
Read [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before changes
or use with model artifacts.

The [original project walkthrough](docs/historical-walkthrough.md) is preserved
verbatim as historical documentation. Its earlier Cassandra/model-selection and
deployment instructions are unsupported. Historical artifacts and stored results
remain in the repository and are unverified under the maintained protocols.
