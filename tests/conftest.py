# tests/conftest.py
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

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
