# Changelog

## Unreleased

- Refresh the official Python container base to an immutable image containing
  Debian APT 3.0.4, which fixes CVE-2026-109518. The image also updates tzdata
  from 2026e to 2026f; Python remains 3.13.18. Rebuild both services and train
  with the updated backend image before serving. Dependency locks, application
  code and benchmark checks are unchanged. Zlib remains at the earlier
  package version; see the [remaining findings](SECURITY.md#container-base-refresh--14-november-2026).

## 0.1.1 — 2026-11-14

- Update the serving lock's transitive `opentelemetry-api` dependency from 1.45.1
  to 1.45.2 for [GHSA-mf5v-w3w3-f6cg](https://github.com/open-telemetry/opentelemetry-python/security/advisories/GHSA-mf5v-w3w3-f6cg),
  which bounds trace-context parsing and removes overlapping regex alternatives.
  All other dependency pins, evaluation protocols and benchmark checks are unchanged.
- Source-checkout users should reinstall the maintained hash-locked requirements
  and reproduce training with a fresh trusted model bundle. This serving-only
  dependency change does not itself invalidate an existing bundle: its manifest
  records evaluation dependencies, not the serving-lock digest. Container users should
  rebuild the backend and train inside that image before serving.

The supported scope remains trusted local reproduction. No exposed Metro attack
path or public-service security guarantee is claimed. Historical results remain
unchanged and 2018 remains unscored.

## 0.1.0 — 2026-11-04

This first supported release uses the source-checkout setup in README.md. The
historical `setup.py` version 0.0.1 and `requirements.txt` are not supported
installation paths; no package-registry distribution is released. Train a fresh
local bundle and retain its manifest digest separately. Regenerate bundles when
source or dependency changes invalidate their pins; do not bypass integrity
checks or reuse the historical pickles.

- Document the maintained source-checkout setup, preserve the original walkthrough
  as historical documentation, and add a technical report linking the frozen
  development results, negative comparisons and reproduction commands.
- Pin and verify Python 3.13.18 before evaluation CI; preserve all existing
  benchmark, container and code-scanning checks.
- Reject detected data, model-parameter and runtime-metadata drift during M1
  evaluation while preserving its original report schema and numerical protocol.


- Update both pinned container bases from Python3.13.17 to 3.13.18 for upstream
  security fixes. Preserve dependency locks, benchmark protocols and container
  restrictions; retain the fresh same-image training/serving check.

- Reject detected dependency-lock, runtime-metadata and protocol changes during
  temporal/weather evaluation. Preserve initial provenance, historical reports
  and numerical protocols; test report failure and competing-output boundaries.

- Reject detected data, prediction-source, dependency-metadata or model-parameter
  changes during local bundle training before creating output. Record the initial
  provenance snapshot and test training, serialization and output failure paths.

- Add a predeclared paired first/last weather-row sensitivity experiment across
  every retained 2017 hour, with 53 fixed expanding-origin folds, quarterly
  diagnostics and raw predictions. Preserve the first-row default, historical
  reports and reserved 2018 isolation.

- Add fixed expanding-window development validation, exact seven-calendar-day
  seasonal baselines, same-cohort predictions and pooled/paired metrics.
- Exclude reserved 2018 values before validation in the new development loader;
  preserve the original loader and report semantics. Add leakage/temporal tests.
- Loader source changes require regenerating local serving bundles; existing
  source-integrity checks continue to reject mismatched bundles.

- Handle malformed JSON, invalid prediction values and network failures in the
  frontend; derive requests from the shared feature contract and report volume
  without unsupported congestion thresholds.
- Reproduce frontend sessions with AppTest and lock its dependencies separately.
- Build non-root Python3.13 containers with explicit model trust, read-only
  mounts, loopback ports and backend readiness; exercise a newly trained model
  and real frontend/API session in container CI.

- Train new local serving bundles with recorded data, code and dependency provenance;
  require a separately trusted manifest pin before loading executable pickle bytes.
- Move API artifact loading from import to configured startup. Share validated
  feature construction with training; derive hour category, reject contradictory
  requests and avoid reflecting raw input or internal errors.
- Add artifact-integrity, startup, request-validation and concurrency tests;
  retain historical models without loading them.

- Add an offline raw-CSV evaluation path with timestamp grouping and fixed
  train/validation/reserved-test boundaries.
- Correct holiday encoding in the new path and explicitly account for invalid
  temperatures and duplicate observations.
- Compare training-only mean/calendar baselines and a seeded forest on the same
  validation cohort, with aggregate/monthly metrics and experiment metadata.
- Add behavioural tests, evaluation dependency hashes and CI.
- Formalize the existing MIT code licence and attribute the UCI dataset under
  its separate CC BY4.0 licence.

Historical database training, saved model artifacts and stored metrics remain
unchanged and unverified. Local serving now uses the explicit fresh-bundle path
described above. The maintained container path trains a fresh bundle in its
backend image; historical models are never substituted.
