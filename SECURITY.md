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
