# Contributing

For evaluation changes, read [the M1 protocol](docs/evaluation.md) and
[the fixed temporal protocol](docs/temporal-evaluation.md) and
[the weather-row sensitivity protocol](docs/weather-sensitivity.md), install
`requirements-serving.txt` and `requirements-frontend.txt` in Python 3.13, and run
`python -m unittest discover -s tests -v` from the repository root.
Keep tests offline and use synthetic fixtures for edge cases.

Explain the observed behavior, expected behavior and verification. Include a
regression test for correctness fixes. Do not tune against the reserved 2018
period or compare metrics from different cohorts as an improvement. Preserve
unfavorable benchmark results and existing historical artifacts. Do not submit
credentials, private datasets or unlicensed material.
