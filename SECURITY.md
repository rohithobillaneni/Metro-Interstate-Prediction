# Security

The offline evaluation reads local CSV and writes a new JSON report. It makes no
network calls and needs no credentials. Use the hash-pinned evaluation lock in
an isolated Python3.13 environment. Dependency metadata checks are not a guarantee
that dependencies are free of vulnerabilities.

Do not load untrusted pickle/joblib files: deserialization can execute code.
The historical model files are not loaded or verified by the offline evaluation.
The legacy API, frontend and Docker setup have not yet received this milestone's
hardening and should not be exposed as a public service.

Please use the repository's GitHub **Security → Report a vulnerability** option
when available. If it is unavailable, open an issue requesting a private contact
channel without disclosing exploit details, credentials or private data.
