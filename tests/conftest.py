# tests/conftest.py
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
import pytest
import docker
import time
import os

# ----------------------------------------------------------------------
# Load .env (non‑secret defaults) so tests can see things like DB host,
# API token, etc.  Secrets are injected via Docker secrets at runtime,
# so they are NOT loaded from the file.
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ----------------------------------------------------------------------
# Fixture: temporary Docker‑Compose stack (engine + db + prometheus)
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def docker_stack():
    """
    Spins up a minimal Docker‑Compose environment for the integration tests.
    The compose file lives at the repository root (docker‑compose.yml) and
    already defines the `engine`, `db`, and `monitor` services.
    """
    compose_file = PROJECT_ROOT / "docker-compose.yml"

    # Bring the stack up (detached)
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # Wait for the engine health endpoint to become ready
    engine_url = "http://localhost:8005/healthz"
    for _ in range(30):
        try:
            r = requests.get(engine_url, timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("Engine health endpoint never became ready")

    yield  # tests run here

    # Tear down after the whole test session
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down", "-v"],
        cwd=PROJECT_ROOT,
    )


# ----------------------------------------------------------------------
# Fixture: a ready‑to‑use HTTP client for the Flask API
# ----------------------------------------------------------------------
@pytest.fixture
def api_client(docker_stack):
    """
    Returns a tiny wrapper around `requests` that automatically injects the
    bearer token (taken from the environment variable set in .env).
    """
    token = os.getenv("CQT_API_TOKEN", "dummy-token-for-tests")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


# ----------------------------------------------------------------------
# Fixture: a clean in‑memory representation of a trade for unit tests
# ----------------------------------------------------------------------
@pytest.fixture
def sample_trade():
    """A minimal dict that mimics the payload accepted by /simulate."""
    return {
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.0800,
        "qty": 0.01,
    }

@pytest.fixture(scope="session")
def docker_client():
    """Return a Docker client that talks to the host Docker daemon."""
    return docker.from_env()

@pytest.fixture(scope="session")
def mt5_sandbox(docker_client):
    """
    Spin up a sandbox MT5 demo container (you must have an image that
    pretends to be an MT5 broker – e.g., `citadel/mt5-demo`).
    The fixture yields the container object and tears it down afterwards.
    """
    image = os.getenv("MT5_SANDBOX_IMAGE", "citadel/mt5-demo:latest")
    container = docker_client.containers.run(
        image,
        detach=True,
        ports={"443/tcp": None},   # expose a random host port
        environment={"DEMO_ACCOUNT": "1"},
    )
    # Wait for the service to be ready (simple health‑check loop)
    host_port = container.attrs["NetworkSettings"]["Ports"]["443/tcp"][0]["HostPort"]
    health_url = f"https://localhost:{host_port}/health"
    for _ in range(30):
        try:
            import urllib3
            http = urllib3.PoolManager(cert_reqs='CERT_NONE')
            resp = http.request("GET", health_url, timeout=2.0)
            if resp.status == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.fail("MT5 sandbox never became healthy")
    yield container, host_port
    container.stop()
    container.remove()

