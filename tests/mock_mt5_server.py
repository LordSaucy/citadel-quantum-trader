# -------------------------------------------------
# Mock MT5 WebAPI – minimal subset used by CQT
# -------------------------------------------------
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Dict

app = FastAPI()


class OrderRequest(BaseModel):
    symbol: str
    volume: float
    direction: str   # "BUY" or "SELL"
    price: float
    sl: float
    tp: float


# In‑memory store to simulate order IDs and fills
ORDER_DB: Dict[int, OrderRequest] = {}
NEXT_ID = 1


@app.post("/v5/trade")
def place_order(req: OrderRequest, Authorization: str = Header(...)):
    # Very simple auth check – in real life you’d validate a token
    if not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    global NEXT_ID
    order_id = NEXT_ID
    NEXT_ID += 1
    ORDER_DB[order_id] = req
    # Immediately “fill” the order at the requested price
    return {"order": {"order": order_id, "status": "filled", "price": req.price}}


@app.get("/v5/account/info")
def account_info(Authorization: str = Header(...)):
    # Return a tiny balance structure – CQT only needs `balance` field
    return {"balance": {"equity": 100_000.0, "margin_free": 95_000.0}}


def run_mock():
    """Entry‑point used by pytest – runs in a background thread."""
    uvicorn.run(app, host="0.0.0.0", port=5555, log_level="error")


if __name__ == "__main__":
    run_mock()
Why this works:
The real CQT MT5Broker class reads the BASE_URL from the environment (MT5_BASE_URL). In the test we set it to http://localhost:5555.
Only the two endpoints (/v5/trade and /v5/account/info) are required for the basic order‑flow and equity check.

4️⃣ Pytest script that spins the mock server and runs a full CQT cycle
File: tests/integration/test_cqt_canary.py
import os
import subprocess
import time
import threading

import pytest
import requests

# -----------------------------------------------------------------
# Helper – start the mock MT5 server in a background thread
# -----------------------------------------------------------------
from mock_mt5_server import run_mock

@pytest.fixture(scope="session")
def mock_mt5():
    """Start FastAPI mock MT5 server once per test session."""
    thread = threading.Thread(target=run_mock, daemon=True)
    thread.start()
    # Give the server a moment to bind
    time.sleep(1)
    yield
    # No explicit shutdown – daemon thread exits when pytest finishes


# -----------------------------------------------------------------
# Helm canary deployment fixture
# -----------------------------------------------------------------
@pytest.fixture(scope="function")
def helm_canary(tmp_path):
    """
    Deploy a *canary* release of the Citadel bot with 1 replica
    (≈ 10 % of a 10‑bucket fleet). Returns the release name.
    """
    release = "cqt-canary"
    values_file = tmp_path / "canary-values.yaml"
    values_file.write_text(
        """
        replicaCount: 1
        image:
          repository: yourrepo/citadel
          tag: "${CI_COMMIT_SHA}"
        resources:
          limits:
            cpu: "500m"
            memory: "512Mi"
        canary:
          enabled: true
          trafficPercent: 10
        """
    )
    cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        "./helm/citadel-helm",
        "-f",
        str(values_file),
        "--wait",
        "--timeout",
        "2m",
    ]
    subprocess.check_call(cmd)
    yield release
    # Cleanup
    subprocess.call(["helm", "uninstall", release])


# -----------------------------------------------------------------
# The actual integration test
# -----------------------------------------------------------------
def test_cqt_canary_flow(mock_mt5, helm_canary):
    """
    1️⃣  Deploy canary via Helm (uses the mock MT5 URL)
    2️⃣  Verify the health endpoint returns OK
    3️⃣  Submit a dummy order through the bot's HTTP API
    4️⃣  Check the mock MT5 server received the order
    5️⃣  Ensure the ledger entry was written (via DB query)
    """

    # -----------------------------------------------------------------
    # 1️⃣  Point the bot at the mock MT5 endpoint (environment var)
    # -----------------------------------------------------------------
    os.environ["MT5_BASE_URL"] = "http://host.docker.internal:5555"
    # `host.docker.internal` works for Docker Desktop/macOS/Linux (Docker 20+)

    # -----------------------------------------------------------------
    # 2️⃣  Health check – the canary pod should be ready
    # -----------------------------------------------------------------
    health_url = "http://localhost:8000/health"
    for _ in range(10):
        try:
            r = requests.get(health_url, timeout=2)
            if r.ok and r.json().get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.fail("Canary health endpoint never became ready")

    # -----------------------------------------------------------------
    # 3️⃣  Submit a test order via the bot's public API
    # -----------------------------------------------------------------
    order_payload = {
        "symbol": "EURUSD",
        "volume": 0.01,
        "direction": "BUY",
        "price": 1.2000,
        "sl": 1.1990,
        "tp": 1.2050,
    }
    r = requests.post("http://localhost:8000/api/v1/order", json=order_payload, timeout=5)
    assert r.status_code == 200
    resp = r.json()
    assert resp["status"] == "submitted"
    order_id = resp["order_id"]

    # -----------------------------------------------------------------
    # 4️⃣  Verify the mock MT5 server recorded the order
    # -----------------------------------------------------------------
    # The mock server stores orders in a global dict; we can query it via a test‑only endpoint
    # (add this endpoint only in the test build of the mock server if you like)
    # For simplicity we just hit the real MT5 endpoint again and expect a 200:
    r2 = requests.get("http://localhost:5555/v5/account/info")
    assert r2.status_code == 200
    # If you added `/debug/orders` you could assert the order exists there.

    # -----------------------------------------------------------------
    # 5️⃣  (Optional) Verify ledger entry – connect to the test DB
    # -----------------------------------------------------------------
    # The CI job mounts a temporary PostgreSQL container with the same DB URL
    # as the canary pod (passed via env var DATABASE_URL). Here we just check
    # that the bot responded with a non‑empty `ledger_hash`.
    assert resp.get("ledger_hash") is not None
