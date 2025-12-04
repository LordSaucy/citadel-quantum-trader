#!/usr/bin/env python3
# -------------------------------------------------
# daily_kpi_monitor.py
# Pulls a few key Prometheus metrics, checks thresholds,
# and posts a nicely‑formatted Slack alert.
# -------------------------------------------------
import requests
from datetime import datetime, timedelta

# -------------------------------------------------
# Configuration – edit these values before deploying
# -------------------------------------------------
PROM_URL = "http://prometheus:9090"          # internal Prometheus address (Docker network)
SLACK_WEBHOOK = "https://hooks.slack.com/services/AAA/BBB/CCC"   # <-- replace!

# -------------------------------------------------
def query(metric: str, lookback_min: int = 15) -> float | None:
    """Return the most recent value of *metric* over the last *lookback_min* minutes."""
    end = datetime.utcnow()
    start = end - timedelta(minutes=lookback_min)

    resp = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={
            "query": metric,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": "60",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()["data"]["result"]
    if not data:
        return None
    # data[0]["values"] is a list of [timestamp, value] strings
    return float(data[0]["values"][-1][1])


def alert(msg: str) -> None:
    """Send a simple payload to the Slack webhook."""
    requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5)


def main() -> None:
    # -------------------------------------------------
    # Pull the four guard‑rail metrics
    # -------------------------------------------------
    wr = query("trading_win_rate")
    dd = query("trading_drawdown_pct")
    lat = query("order_latency_seconds")
    rej = query("order_reject_total")

    alerts = []

    if wr is not None and wr < 0.92:
        alerts.append(f"*⚠️ Win‑rate low*: {wr:.1%}")

    if dd is not None and dd > 0.15:
        alerts.append(f"*⚠️ Draw‑down high*: {dd:.1%}")

    if lat is not None and lat > 0.25:
        alerts.append(f"*⚠️ Latency high*: {lat:.2f}s")

    if rej is not None and rej > 0.02:
        alerts.append(f"*⚠️ Rejection rate*: {rej:.2%}")

    if alerts:
        alert_msg = (
            "*Citadel Quantum Trader – Daily Guardrails Triggered*\n"
            + "\n".join(alerts)
        )
        alert(alert_msg)

        # -----------------------------------------------------------------
        # OPTIONAL: write a flag to the DB so the bot can pause new entries.
        # You need a DB client library (psycopg2, sqlite3, etc.) and the
        # connection string.  Below is a generic placeholder.
        # -----------------------------------------------------------------
        # import psycopg2
        # conn = psycopg2.connect(dsn=os.getenv("DATABASE_URL"))
        # cur = conn.cursor()
        # cur.execute(
        #     "INSERT INTO system_flags (name, value) VALUES ('kill_switch', 1) "
        #     "ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value"
        # )
        # conn.commit()
        # cur.close()
        # conn.close()
        # -----------------------------------------------------------------
        # Comment out the block above if you don’t want an automatic kill‑switch.
        # -----------------------------------------------------------------


if __name__ == "__main__":
    main()
