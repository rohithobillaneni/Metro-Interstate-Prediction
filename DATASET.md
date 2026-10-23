# Dataset provenance and licence

The bundled `data/Metro_Interstate_Traffic_Volume.csv` contains **Metro Interstate
Traffic Volume**, donated by John Hogue (2019), UCI Machine Learning Repository:
[dataset record](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume),
[DOI 10.24432/C5X60B](https://doi.org/10.24432/C5X60B).
It is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
separately from this repository's MIT source-code licence.

The data describes hourly westbound I-94 traffic at Minnesota DoT station 301,
with weather and holiday attributes. It contains public aggregate counts, not
individual vehicle records. No private or employer dataset is required.

On 2026-10-23 the bundled file's parsed CSV rows and ordering were verified
against the [UCI download](https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip).
The CSV bytes differ only in line endings. UCI does not expose a dataset version
number here; identify this snapshot by its DOI and hashes:

| Artifact | SHA-256 |
|---|---|
| Bundled CSV | `0411ba6c43127a29cefa4d8f39708b9d77979d9dcf895f2d212506fe947b673c` |
| UCI ZIP | `b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768` |
| UCI decompressed CSV | `749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e` |

The raw file has 48,204 rows. The offline evaluation converts Kelvin to Celsius,
excludes ten nonpositive-Kelvin rows, groups repeated timestamps, and derives
calendar features. See [methodology](docs/evaluation.md) for the exact policy.
The processed CSVs already present in `data/` are historical derivatives; their
full production path is not reconstructed and they are not used by the new
evaluation. This notice attributes those derivatives to the same UCI source
without asserting their transformations or results are correct.

Redistributors must retain attribution, the licence link and an indication of
changes. Neither the dataset donor nor UCI endorses this implementation.
