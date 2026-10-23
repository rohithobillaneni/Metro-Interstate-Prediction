# Changelog

## Unreleased

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
described above; frontend startup and Docker reproduction remain follow-up work.
