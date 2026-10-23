"""Build, train and exercise an isolated local stack; never publish an image."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
REQUEST = {
    "temp": 15, "clouds_all": 50, "holiday": "No", "weather_main": "Clouds",
    "weekday": "Monday", "hour": 12, "month": "January",
}


def main() -> None:
    project = "metro-smoke-" + uuid.uuid4().hex[:12]
    environment = {**os.environ, "COMPOSE_PROJECT_NAME": project,
                   "METRO_BACKEND_PORT": "0", "METRO_FRONTEND_PORT": "0",
                   "METRO_MODEL_MANIFEST_SHA256": ""}
    compose = ["docker", "compose", "--project-name", project,
               "--file", str(ROOT / "docker-compose.yml")]
    images = [f"{project}-{service}:local" for service in ("backend", "frontend")]
    trainer = f"{project}-training"

    def run(command, *, capture=False, input=None, check=True, timeout=600):
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                capture_output=capture, input=input, timeout=timeout)
        if check and result.returncode:
            if capture:
                print(result.stdout)
                print(result.stderr)
            result.check_returncode()
        return result

    # No retained service, image tag or model directory is reused.
    with tempfile.TemporaryDirectory(prefix=project + "-") as temporary:
        output = Path(temporary)
        environment["METRO_MODEL_HOST_DIR"] = str(output / "bundle")
        try:
            run([*compose, "build"], timeout=900)
            training = run([
                "docker", "run", "--rm", "--name", trainer, "--network", "none",
                "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", f"type=bind,source={ROOT / 'data/Metro_Interstate_Traffic_Volume.csv'},target=/input.csv,readonly",
                "--mount", f"type=bind,source={output},target=/output",
                images[0], "python", "-m", "src.artifacts", "--data", "/input.csv",
                "--output", "/output/bundle",
            ], capture=True)
            # Pin comes from this reviewed local training command, never sidecar discovery.
            pin = json.loads(training.stdout)["manifest_sha256"]
            environment["METRO_MODEL_MANIFEST_SHA256"] = pin
            run([*compose, "up", "--detach", "--wait", "--wait-timeout", "120"], timeout=180)

            def address(service, port):
                published = run([*compose, "port", service, str(port)], capture=True).stdout.strip()
                if not published.startswith("127.0.0.1:"):
                    raise RuntimeError("Smoke stack must publish only to loopback")
                return "http://" + published

            backend, frontend = address("backend", 8000), address("frontend", 8501)
            with urllib.request.urlopen(frontend + "/_stcore/health", timeout=10) as response:
                assert response.status == 200
            request = urllib.request.Request(backend + "/predict", json.dumps(REQUEST).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                prediction = json.load(response)["prediction"]
            assert type(prediction) in (int, float) and math.isfinite(prediction) and prediction >= 0
            invalid = urllib.request.Request(backend + "/predict", b'{"hour": true}',
                                             {"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(invalid, timeout=10)
            except urllib.error.HTTPError as exc:
                assert exc.code == 422
                exc.close()
            else:
                raise AssertionError("Invalid request was accepted")
            # Health alone does not run a Streamlit session. Exercise its real API request.
            session = run([*compose, "exec", "-T", "frontend", "python", "-"], capture=True, input='''
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("frontend.py").run(timeout=30)
assert not app.exception, app.exception
app.button[0].click().run(timeout=30)
assert not app.exception, app.exception
assert not app.error, app.error
assert len(app.success) == 1, app.success
print(app.success[0].value)
''')
            # A wrong independently supplied pin must not reach deserialization/readiness.
            wrong_pin = run([*compose, "run", "--rm", "--no-deps", "-e",
                             "METRO_MODEL_MANIFEST_SHA256=" + "0" * 64, "backend",
                             "python", "-c", "from src.artifacts import load_bundle; load_bundle('/models', '0'*64)"],
                            capture=True, check=False, timeout=60)
            assert wrong_pin.returncode != 0 and "separately trusted pin" in wrong_pin.stderr
            print(json.dumps({"project": project, "prediction": prediction,
                              "frontend_session": session.stdout.strip(),
                              "manifest_sha256": pin, "invalid_request_status": 422,
                              "wrong_pin_rejected": True, "platform": "current Docker engine only"}))
        finally:
            # Capture diagnostics before removing only resources owned by this invocation.
            original_failure = sys.exc_info()[0] is not None
            cleanup_errors = []
            cleanup = [
                ([*compose, "logs", "--no-color"], False, 30),
                (["docker", "rm", "--force", trainer], False, 30),
                ([*compose, "down", "--volumes", "--remove-orphans"], True, 60),
                (["docker", "image", "rm", *images], not original_failure, 60),
            ]
            for command, required, timeout in cleanup:
                try:
                    run(command, check=required, timeout=timeout)
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_errors.append(str(exc))
            if cleanup_errors:
                print("Cleanup diagnostics:", cleanup_errors, file=sys.stderr)
                if not original_failure:
                    raise RuntimeError("Smoke resources could not all be cleaned up")


if __name__ == "__main__":
    main()
