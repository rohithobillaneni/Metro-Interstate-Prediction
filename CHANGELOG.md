# Changelog

## Unreleased

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
