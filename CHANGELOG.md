# Changelog

## Unreleased

- Add an offline raw-CSV evaluation path with timestamp grouping and fixed
  train/validation/reserved-test boundaries.
- Correct holiday encoding in the new path and explicitly account for invalid
  temperatures and duplicate observations.
- Compare training-only mean/calendar baselines and a seeded forest on the same
  validation cohort, with aggregate/monthly metrics and experiment metadata.
- Add behavioural tests, evaluation dependency hashes and CI.
- Formalize the existing MIT code licence and attribute the UCI dataset under
  its separate CC BY4.0 licence.

Legacy serving, database training, model artifacts and stored metrics remain
unchanged and are not certified by this evaluation milestone.
