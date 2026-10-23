# Offline evaluation architecture

The maintained evaluation path serves engineers who want to reproduce and audit
this traffic-estimation example locally. It requires no database, API key,
network service or saved model. It does not claim advance forecasting or deploy
an operational traffic system.

1. `src.offline_data.load_csv` reads the public raw CSV, validates the feature
   contract and target agreement, and creates immutable timestamped records.
2. `partition_records` applies fixed calendar cutoffs to unique timestamps.
3. `src.evaluation` fits two baselines and a seeded random forest on training
   records only. The model reuses `get_preprocessing_pipeline` from the original
   implementation. Scaling and category discovery occur within that fit.
4. Validation predictions produce same-cohort aggregate and monthly metrics in
   a JSON report with dataset digest, exclusions, configuration and environment.

The CLI is `python -m src.evaluation --data PATH [--output NEW_PATH]`. Invalid
CSV/configuration and I/O failures return a nonzero exit with a diagnostic.
Existing report files are never overwritten. No pickle is read or written.
The library functions are internal project interfaces, not a stable published
package API. Calendar cutoffs are fixed in the CLI to prevent silent protocol
changes; library partitioning accepts explicit cutoffs for tests and future work.

For this small dataset, records and pandas features are materialized in memory.
The forest uses one worker, 100 trees, depth at most 12 and a fixed random seed.
No plugin architecture or service abstraction is needed for this milestone.

The Cassandra ingestion/training scripts, saved pickle models, FastAPI/Streamlit
serving path and Docker configuration are retained as historical components.
They have not yet been made reproducible by this change. They are not imported
by the offline command. In particular, do not load an untrusted pickle: loading
one can execute code. Future serving work must use a newly generated and
traceable artifact rather than assume the historical artifacts are compatible.
