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
Use loopback binding. Authentication, public deployment, frontend startup and
the legacy Docker setup remain outside the verified scope.

Please use the repository's GitHub **Security → Report a vulnerability** option
when available. If it is unavailable, open an issue requesting a private contact
channel without disclosing exploit details, credentials or private data.
