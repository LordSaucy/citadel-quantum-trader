run_paper_trading.py

Launches the Citadel bot in paper‑trading mode, monitors latency,
rejection‑rate and slippage, and stops after 48 h or ≥ 500 trades.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# -----------------------------------------------------------------
# Configuration (adjust if you use a different Prometheus port)
# -----------------------------------------------------------------
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
BOT_HEALTH_URL = "http://localhost:8000/health"
METRIC_POLL_INTERVAL = 30          # seconds
MAX_TRADE_COUNT = 500
MAX_DURATION = timedelta(hours=48)  # 48 h

# -----------------------------------------------------------------
# Helper – query a single Prometheus expression
# -----------------------------------------------------------------
def prom_query(expr: str) -> float:
    resp = requests.get(PROMETHEUS_URL, params={"query": expr})
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success" or not data["data"]["result"]:
        return 0.0
    # Take the first sample (most recent)
    return float(data["data"]["result"][0]["value"][1])


# -----------------------------------------------------------------
# Helper – wait for the bot to become healthy
# -----------------------------------------------------------------
def wait_for_bot(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(BOT_HEALTH_URL, timeout=5)
            if r.ok and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# -----------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a 48 h (or 500‑trade) paper‑trading campaign"
    )
    parser.add_argument(
        "--compose",
        default="docker-compose.yml",
        help="Base compose file (default: docker-compose.yml)",
    )
    parser.add_argument(
        "--override",
        default="docker-compose.paper.yml",
        help="Paper‑trading override file",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1️⃣  Start the Docker stack (paper mode)
    # -------------------------------------------------------------
    print("🚀 Starting paper‑trading stack …")
    up_cmd = [
        "docker",
        "compose",
        "-f",
        args.compose,
        "-f",
        args.override,
        "up",
        "-d",
    ]
    subprocess.check_call(up_cmd)

    # -------------------------------------------------------------
    # 2️⃣  Wait for health endpoint
    # -------------------------------------------------------------
    print("⏳ Waiting for bot health …")
    if not wait_for_bot():
        print("❌ Bot never became healthy – aborting", file=sys.stderr)
        subprocess.run(
            ["docker", "compose", "-f", args.compose, "-f", args.override, "down"]
        )
        return 1

    print("✅ Bot is healthy – monitoring metrics …")

    # -------------------------------------------------------------
    # 3️⃣  Monitoring loop
    # -------------------------------------------------------------
    start_time = datetime.utcnow()
    trade_counter = 0
    max_latency = 0.0
    total_rejects = 0
    total_slip = 0.0
    slip_samples = 0

    while True:
        # ---- latency (seconds) – we take the max observed over the interval
        latency = prom_query('max(cqt_latency_seconds)')
        max_latency = max(max_latency, latency)

        # ---- rejection rate (total count)
        rejects = prom_query('sum(cqt_reject_total)')
        total_rejects = int(rejects)

        # ---- slippage (pips) – average over the interval
        slip = prom_query('avg(cqt_slippage_pips)')
        if slip > 0:
            total_slip += slip
            slip_samples += 1

        # ---- trade count – we can approximate from the reject counter + fills
        # (the validator increments a Prometheus counter `cqt_trade_total` on every
        #  successful fill; if you haven't added it, you can derive it from the
        #  ledger CSV later – here we just use a rough estimate)
        trade_counter = int(prom_query('sum(cqt_trade_total)'))

        # ---------------------------------------------------------
        # 4️⃣  Check termination conditions
        # ---------------------------------------------------------
        elapsed = datetime.utcnow() - start_time
        if trade_counter >= MAX_TRADE_COUNT:
            print(f"🏁 Reached {trade_counter} trades – stopping")
            break
        if elapsed >= MAX_DURATION:
            print(f"⌛ 48 h elapsed ({elapsed}) – stopping")
            break

        # ---------------------------------------------------------
        # 5️⃣  Sleep until next poll
        # ---------------------------------------------------------
        time.sleep(METRIC_POLL_INTERVAL)

    # -------------------------------------------------------------
    # 6️⃣  Shut down the stack
    # -------------------------------------------------------------
    print("🛑 Stopping Docker stack …")
    subprocess.run(
        ["docker", "compose", "-f", args.compose, "-f", args.override, "down"]
    )

    # -------------------------------------------------------------
    # 7️⃣  Pull the ledger CSV (exported by the validator)
    # -------------------------------------------------------------
    # The validator writes `backtest_trades.csv` in the working dir.
    # If you use a different path, adjust accordingly.
    csv_path = Path("backtest_trades.csv")
    if not csv_path.is_file():
        print("⚠ No trade CSV found – you may need to enable export_trades_to_csv()", file=sys.stderr)

    # -------------------------------------------------------------
    # 8️⃣  Compute final statistics
    # -------------------------------------------------------------
    avg_slippage = total_slip / slip_samples if slip_samples else 0.0
    reject_rate = total_rejects / trade_counter if trade_counter else 0.0

    print("\n===== PAPER‑TRADING SUMMARY =====")
    print(f"Duration                : {elapsed}")
    print(f"Total trades executed   : {trade_counter}")
    print(f"Max observed latency    : {max_latency:.3f} s")
    print(f"Overall reject rate     : {reject_rate*100:.2f}%")
    print(f"Average slippage        : {avg_slippage:.3f} pips")
    print("\n--- PASS / FAIL criteria
