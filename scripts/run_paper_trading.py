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

      # -----------------------------------------------------------------
    # 9️⃣  Apply the production tolerances
    # -----------------------------------------------------------------
    # These numbers are the same limits you enforce in live trading.
    # Adjust them here if your risk team changes the policy.
    LATENCY_TOLERANCE_SEC   = 0.5      # ≤ 0.5 seconds per order
    REJECT_RATE_TOLERANCE   = 0.01     # ≤ 1 % of all orders rejected
    SLIPPAGE_TOLERANCE_PIPS = 0.5      # ≤ 0.5 pips average slippage

    failures = []

    if max_latency > LATENCY_TOLERANCE_SEC:
        failures.append(
            f"❌ LATENCY EXCEEDED – max {max_latency:.3f}s > {LATENCY_TOLERANCE_SEC}s"
        )
    else:
        print(f"✅ Latency within tolerance (≤ {LATENCY_TOLERANCE_SEC}s)")

    if reject_rate > REJECT_RATE_TOLERANCE:
        failures.append(
            f"❌ REJECTION RATE EXCEEDED – {reject_rate*100:.2f}% > {REJECT_RATE_TOLERANCE*100:.2f}%"
        )
    else:
        print(f"✅ Rejection rate within tolerance (≤ {REJECT_RATE_TOLERANCE*100:.2f}%)")

    if avg_slippage > SLIPPAGE_TOLERANCE_PIPS:
        failures.append(
            f"❌ SLIPPAGE EXCEEDED – avg {avg_slippage:.3f} pips > {SLIPPAGE_TOLERANCE_PIPS} pips"
        )
    else:
        print(f"✅ Slippage within tolerance (≤ {SLIPPAGE_TOLERANCE_PIPS} pips)")

    # -----------------------------------------------------------------
    # 10️⃣  Emit a concise result for CI / human consumption
    # -----------------------------------------------------------------
    if failures:
        print("\n=== PAPER‑TRADING RESULT: **FAIL** ===")
        for f in failures:
            print(f)
        # Return a non‑zero exit code so CI marks the job as failed
        return 1
    else:
        print("\n=== PAPER‑TRADING RESULT: **PASS** ===")
        print("All metrics satisfied the production tolerances.")
        return 0

# -------------------------------------------------
# 11️⃣  PAPER‑TRADING GATE (48 h or 500 trades)
# -------------------------------------------------
paper-trading:
  name: Paper‑Trading Gate
  runs-on: ubuntu-latest
  needs: [build]                     # wait until the Docker image is built
  timeout-minutes: 1800              # 30 h max (covers 48 h + buffer)
  steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Docker
      uses: docker/setup-buildx-action@v2

    - name: Pull built image
      run: |
        docker pull ghcr.io/${{ github.repository_owner }}/citadel/trader:latest

    - name: Start paper stack (detached)
      run: |
        docker compose -f docker-compose.yml -f docker-compose.paper.yml up -d

    - name: Run paper‑trading monitor (48 h or 500 trades)
      env:
        LATENCY_TOLERANCE_SEC: "0.5"
        REJECT_TOLERANCE: "0.01"
        SLIPPAGE_TOLERANCE_PIPS: "0.5"
      run: |
        python scripts/run_paper_trading.py

    - name: Collect artefacts
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: paper‑run‑artifacts
        path: |
          backtest_trades.csv
          paper_summary.txt
          metrics_snapshot.json
