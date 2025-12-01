# tests/test_integration.py
import json
import time

import pytest


# ----------------------------------------------------------------------
# Helper: wait until Prometheus has scraped the engine at least once
# ----------------------------------------------------------------------
def wait_for_metric(api_client, metric_name, timeout=10):
    """Poll the /metrics endpoint until the given metric appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = api_client.get("http://localhost:8000/metrics")
        if resp.status_code == 200 and metric_name.encode() in resp.content:
            return True
        time.sleep(0.5)
    raise TimeoutError(f"Metric {metric_name} never appeared")


def test_engine_health_endpoint(api_client):
    """The Flask health endpoint must return 200 OK."""
    r = api_client.get("http://localhost:8005/healthz")
    assert r.status_code == 200
    assert r.text.strip().lower() == "ok"


def test_simulate_trade_flow(api_client, sample_trade):
    """Submit a paper‑trade, verify DB row and Prometheus metric."""
    # 1️⃣  POST /simulate
    r = api_client.post(
        "http://localhost:8005/simulate",
        json=sample_trade,
        timeout=5,
    )
    assert r.status_code == 200
    payload = r.json()
    assert "trade_id" in payload
    trade_id = payload["trade_id"]

    # 2️⃣  Verify the trade landed in TimescaleDB
    # (We use the DB container directly – no external network)
    db_check = (
        "docker exec timescaledb "
        "psql -U cqt_user -d cqt_ledger -t -c "
        f\"SELECT trade_id FROM trades WHERE trade_id={trade_id};\""
    )
    result = subprocess.check_output(db_check, shell=True, text=True).strip()
    assert result == str(trade_id)

    # 3️⃣  Verify Prometheus metric incremented
    # Wait for the engine to expose the updated metric
    wait_for_metric(api_client, "cqt_orders_total")
    metrics = api_client.get("http://localhost:8000/metrics").text
    # Look for a line like: cqt_orders_total{symbol="EURUSD"} 1
    assert f'cqt_orders_total{{symbol="{sample_trade["symbol"]}"}}' in metrics


def test_prometheus_scrapes_engine(api_client):
    """Prometheus must be able to scrape the engine /metrics endpoint."""
    # The monitor service is reachable on localhost:9090 inside the test runner
    prom_resp = api_client.get("http://localhost:9090/api/v1/targets")
    assert prom_resp.status_code == 200
    data = prom_resp.json()
    # Find the target whose job is "cqt_engine"
    engine_targets = [
        t for t in data["data"]["activeTargets"] if t["labels"]["job"] == "cqt_engine"
    ]
    assert len(engine_targets) == 2  # primary + standby
    for tgt in engine_targets:
        assert tgt["health"] == "up"
