# v0.1.0 security assessment — 4 November 2026

This dated assessment covers the maintained source-checkout and local container
workflow. It is not an assurance for public deployment. Follow [SECURITY.md](../SECURITY.md)
and the [model trust boundary](serving.md#trust-boundary).

## Inputs and checks

Both Dockerfiles retain the official Python 3.13.18 image index
`sha256:b3575dcf816cdacc5292df77a6e3e59b8941631078373a045efa2ac1e9c5c3b2`.
The previously retrieved amd64/arm64 manifest, configuration and layer bytes
were rehashed against that immutable index. Their installed Debian 13 inventories
contain 87 binary packages and 61 source/version pairs per architecture. This
metadata inspection does not establish arm64 runtime behavior.

The current `python:3.13-slim` index was also inspected:
`sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1`.
Its amd64/arm64 images record the same Python source version/hash, bundled pip
version and installed Debian package versions. That comparison identifies no
package-version remedy for these findings; it does not establish binary
equivalence or prove the absence of other changes. The release retains its pin.

On this date, 114 [OSV batch queries](https://google.github.io/osv.dev/post-v1-querybatch/)
covered those 61 Debian source pairs, all 51 package/version pairs in the three
maintained Python locks, the image's bundled pip, and the CPython 3.13.18 source
commit `0d4bd512fc429e3d4eacdd113c975e161f21644d`. All responses were present, without
errors or pending pagination. The Python package pairs, pip and CPython commit
returned no matches. This does not rule out unknown vulnerabilities, vendored
native-library issues or defects missed by the advisory sources.

The Debian queries returned 84 advisory IDs. The
[Debian tracker](https://security-tracker.debian.org/tracker/) classified 81 as
open, two as undetermined and one as resolved/not affected. The latter is
[CVE-2026-39113](https://security-tracker.debian.org/tracker/CVE-2026-39113):
the tracker records `fixed_version: 0`, while OSV still matches the source version.
These are source-package matches, not 84 demonstrated exploitable Metro paths.

## Residual findings

The earlier 69 matches retain their dispositions and limitations. They include
privileged command-line operations and services not exposed by the maintained
stack, specialist native interfaces, and Debian-designated minor/unimportant
issues. Glibc DNS configuration failures remain an availability risk. The PAM
classification remains undetermined. Native C++ and compression paths have not
received exhaustive binary call-graph analysis.

Fifteen additional matches were present on 4 November:

| Component | Findings | Supported-scope assessment |
|---|---|---|
| GCC 14 / libstdc++ | CVE-2026-114690, CVE-2026-114691, CVE-2026-115455 | Parallel search, oversized boolean-vector allocation and wide-character bitset extraction failures. Metro exposes bounded scalar prediction fields and trusted local training data, not these interfaces. Native dependencies may use C++; absence of a demonstrated path is not a binary-level proof of unreachability. Debian marks these minor with no fixed trixie version recorded. |
| Glibc | CVE-2026-84243 | Exploitation requires attacker-controlled locale environment and a crafted message catalogue across a trust boundary. Runtime configuration and local files are trusted here; the API has no environment or catalogue upload interface. This disposition does not cover hostile shared-host configuration. Debian marks it minor with no fixed trixie version recorded. |
| SQLite | CVE-2026-119371 through CVE-2026-119379, CVE-2026-119586, CVE-2026-119591 | Crafted SQL, database/WAL files, FTS/R-Tree/session operations or shell recovery input. The maintained workflow accepts CSV and validated prediction fields; it exposes no SQL/database upload or recovery operation. The shell recovery finding remains undetermined in Debian; the others are marked minor without a fixed trixie version recorded. |

The inspected application paths do not establish a critical exploit within this
limited contract. This is a reachability assessment, not a claim that the image
is free of vulnerabilities. The assessment must be revisited when the supported
interfaces, dependency locks, base image or upstream fixes change. Advisory
counts are dated and must not be presented as permanent security guarantees.

## Operating boundary

Use a patched Python interpreter, reviewed source, trusted local CSV and freshly
trained bundles. Pickle remains executable trusted material. A manifest checksum
does not authenticate an artifact's author or sandbox it. Bind to loopback;
non-root users, read-only mounts, dropped capabilities and disabled telemetry
are useful restrictions, not authentication or general adversarial isolation.
The host, container engine and same-user filesystem access remain trusted.

No new dependency, benchmark or runtime pin was introduced for this release.
Keep [historical artifacts](historical-walkthrough.md) separate from the supported
workflow; their security and results are not validated by this assessment.
