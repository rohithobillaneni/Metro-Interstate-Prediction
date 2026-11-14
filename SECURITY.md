# Security

The offline evaluation reads local CSV and writes a new JSON report. It makes no
network calls and needs no credentials. Use the hash-pinned evaluation lock in
an isolated Python3.13 environment. Dependency metadata checks are not a guarantee
that dependencies are free of vulnerabilities.

Do not load untrusted pickle/joblib files: deserialization can execute code.
The historical model files are not loaded or verified. Local API serving now
requires a newly generated bundle and a separately trusted manifest digest.
The digest binds model bytes, source and dependencies; it does not make an
untrusted pickle safe. Read [the trust boundary](docs/serving.md) before use.
Use loopback binding. The supported containers run as non-root users, mount the
model read-only, drop capabilities and disable Streamlit usage telemetry. These
controls are not authentication: public deployment remains outside this scope.
Container base-image digests and dependency locks need deliberate security updates.
For source-checkout use, install a security-maintained Python3.13 patch release
(at least 3.13.18 as of 27 October2026). Dependency locks do not patch the Python
interpreter or operating system. The containers pin that interpreter update;
known Debian base-package advisory matches remain under review. This is not a
claim that the full image is free of vulnerabilities.

Please use the repository's GitHub **Security → Report a vulnerability** option
when available. If it is unavailable, open an issue requesting a private contact
channel without disclosing exploit details, credentials or private data.

The [v0.1.0 dated security assessment](docs/release-security-0.1.0.md) records
the checked inputs, known Debian matches and supported-scope limitations.

## OpenTelemetry dependency update — 14 November 2026

The v0.1.0 serving lock includes `opentelemetry-api` 1.45.1, affected by
[GHSA-mf5v-w3w3-f6cg](https://github.com/open-telemetry/opentelemetry-python/security/advisories/GHSA-mf5v-w3w3-f6cg).
Version 0.1.1 pins 1.45.2, which limits `traceparent` length before parsing and
removes overlapping regex alternatives. The upstream issue requires untrusted
trace-context input that preserves embedded newlines; this update does not
establish that the maintained Metro routes expose that input path.

Upgrade the source checkout, reinstall its hash-locked requirements and regenerate
the trusted model bundle. Rebuild containers and train in the updated backend
image. See the [migration notes](CHANGELOG.md#011--2026-11-14). The earlier dated
assessment remains historical: this dependency patch does not resolve the
inherited Debian findings or extend support to public deployment.

## Container base refresh — 14 November 2026

The base refresh after v0.1.1 pins official Python image index
`sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90`.
Verified amd64/arm64 layer metadata contains APT and libapt-pkg7.0 3.0.4,
which [Debian identifies as fixed for CVE-2026-109518](https://security-tracker.debian.org/tracker/CVE-2026-109518).
The advisory concerns manipulated APT package downloads; the maintained
Dockerfiles and application do not invoke APT. This removes an affected
inherited package version without claiming an exposed Metro exploit.

Both inventories contain 87 Debian binary packages. The only version changes
are those two APT binaries and tzdata 2026e to 2026f. Python 3.13.18, its source
hash and bundled pip 26.2.2 remain unchanged. Metadata inspection for two
architectures is distinct from runtime verification by the Linux container CI.

The dated advisory query still returns 94 Debian source-package matches.
Zlib remains `1:1.3.dfsg+really1.3.1-1` (binary rebuild `+b1`), below Debian's
fixed `1:1.3.dfsg+really1.3.1-1+deb13u1`. In particular,
[CVE-2026-85091](https://security-tracker.debian.org/tracker/CVE-2026-85091)
can follow any failed gzip write; on this older version a subsequent
`gzclearerr` call is required. A nonblocking-only explanation is insufficient.
Native/transitive reachability is not exhaustively established. The tracker
also maps [CVE-2026-76844](https://security-tracker.debian.org/tracker/CVE-2026-76844)
to zlib while describing webpack-dev-middleware; that discrepancy remains
unresolved and is not dismissed. A tracker status of resolved does not mean
the fixed package is installed in this image.

The supported scope remains trusted local reproduction. Further zlib maintenance
is pending; this is not a clean-image or public-service security guarantee.
Published v0.1.1 is unchanged. The refresh is included in
[v0.1.2](CHANGELOG.md#012--2026-11-14); rebuild both services and train a fresh
bundle in the updated backend image before serving.
