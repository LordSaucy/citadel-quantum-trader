#!/usr/bin/env python3
"""
run_shadow.py

Launches the Citadel stack in SHADOW mode, runs for a configurable
duration (or until a trade count is reached), stops the stack,
and then compares the shadow log against the paper‑trading log.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# -----------------------------------------------------------------
# Configuration (adjust as needed)
# -----------------------------------------------------------------
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
BOT_HEALTH_URL = "http://localhost:8000/health"
METRIC_POLL_INTERVAL = 30          # seconds
MAX_TRADE_COUNT = 500
MAX_DURATION = timedelta(hours=48)  # 48 h

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

def prom_query(expr: str) -> float:
    resp = requests.get(PROMETHEUS_URL, params={"query": expr})
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success" or not data["data"]["result"]:
        return 0.0
    return float(data["data"]["result"][0]["value"][1])

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Shadow (live‑mirror, no‑capital) campaign"
    )
    parser.add_argument(
        "--compose",
        default="docker-compose.yml",
        help="Base compose file (default: docker-compose.yml)",
    )
    parser.add_argument(
        "--override",
        default="docker-compose.shadow.yml",
        help="Shadow‑mode override file",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1️⃣  Start the Docker stack (shadow mode)
    # -------------------------------------------------------------
    print("🚀 Starting Shadow stack …")
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
    # 3️⃣  Monitoring loop (same as paper‑trading)
    # -------------------------------------------------------------
    start_time = datetime.now()
    trade_counter = 0
    max_latency = 0.0
    total_rejects = 0
    total_slip = 0.0
    slip_samples = 0

    while True:
        latency = prom_query('max(cqt_order_latency_seconds{shadow="yes"})')
        max_latency = max(max_latency, latency)

        rejects = prom_query('sum(cqt_orders_total{shadow="yes", success="false"})')
        total_rejects = int(rejects)

        slip = prom_query('avg(cqt_order_slippage_pips{shadow="yes"})')
        if slip > 0:
            total_slip += slip
            slip_samples += 1

        trade_counter = int(prom_query('sum(cqt_orders_total{shadow="yes"})'))

        elapsed = datetime.now() - start_time
        if trade_counter >= MAX_TRADE_COUNT:
            print(f"🏁 Reached {trade_counter} shadow trades – stopping")
            break
        if elapsed >= MAX_DURATION:
            print(f"⌛ 48 h elapsed ({elapsed}) – stopping")
            break

        time.sleep(METRIC_POLL_INTERVAL)

    # -------------------------------------------------------------
    # 4️⃣  Shut down the stack
    # -------------------------------------------------------------
    print("🛑 Stopping Docker stack …")
    subprocess.run(
        ["docker", "compose", "-f", args.compose, "-f", args.override, "down"]
    )

    # -------------------------------------------------------------
      # -------------------------------------------------------------
    # 5️⃣  Pull the shadow log (mounted inside the container at
    #      /var/log/cqt_shadow.log) and compare it with the
    #      paper‑trading log (mounted at /var/log/cqt_paper.log)
    # -------------------------------------------------------------
    print("📂 Retrieving shadow log …")
    shadow_log_host_path = Path("/tmp/cqt_shadow.log")
    paper_log_host_path  = Path("/tmp/cqt_paper.log")

    # The Docker‑Compose file mounts the host directory `./logs` into the
    # container at `/var/log`.  We therefore copy the files from the
    # container to a temporary location on the host.
    try:
        # Shadow container name is `cqt-engine` (same as in compose)
        subprocess.check_call([
            "docker", "cp",
            "cqt-engine:/var/log/cqt_shadow.log",
            str(shadow_log_host_path)
        ])
        subprocess.check_call([
            "docker", "cp",
            "cqt-engine:/var/log/cqt_paper.log",
            str(paper_log_host_path)
        ])
    except subprocess.CalledProcessError as exc:
        print(f"⚠️  Could not retrieve logs: {exc}", file=sys.stderr)
        return 2

    # -------------------------------------------------------------
    # 6️⃣  Load both logs, turn them into DataFrames and compute stats
    # -------------------------------------------------------------
    print("🔎 Analysing logs …")
    try:
        shadow_df = process_log(shadow_log_host_path)
        paper_df  = process_log(paper_log_host_path)
    except Exception as exc:
        print(f"❌ Failed to parse logs: {exc}", file=sys.stderr)
        return 3

    # Basic sanity check – both logs should contain the same columns
    common_cols = set(shadow_df.columns) & set(paper_df.columns)
    if not common_cols:
        print("⚠️  No overlapping columns between logs – cannot compare.", file=sys.stderr)
        return 4

    # -----------------------------------------------------------------
    # Helper to compute a few key metrics from a DataFrame
    # -----------------------------------------------------------------
    def summarize(df: pd.DataFrame, label: str) -> Dict[str, float]:
        """Return a dict of aggregated metrics for printing."""
        # We assume the log contains at least:
        #   - timestamp (ISO string)
        #   - latency_seconds (float)
        #   - success (bool/int)
        #   - slippage_pips (float, optional)
        #   - reject_reason (string, optional)
        out = {}
        if "latency_seconds" in df:
            out["avg_latency"] = df["latency_seconds"].mean()
            out["max_latency"] = df["latency_seconds"].max()
        if "success" in df:
            successes = df["success"].astype(bool).sum()
            total     = len(df)
            out["win_rate"] = successes / total * 100 if total else 0.0
        if "slippage_pips" in df:
            out["avg_slip"] = df["slippage_pips"].mean()
        if "reject_reason" in df:
            out["rejects"] = df["reject_reason"].notna().sum()
        # Add a label for pretty printing
        out["label"] = label
        return out

    shadow_stats = summarize(shadow_df, "SHADOW")
    paper_stats  = summarize(paper_df,  "PAPER ")

    # -----------------------------------------------------------------
    # Pretty‑print a side‑by‑side comparison table
    # -----------------------------------------------------------------
    def fmt(v: Any) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

    headers = ["Metric", "Shadow", "Paper", "Δ (Shadow‑Paper)"]
    rows = []
    metric_keys = set(shadow_stats) | set(paper_stats)
    metric_keys.discard("label")   # we already know the labels

    for key in sorted(metric_keys):
        s_val = shadow_stats.get(key, 0.0)
        p_val = paper_stats.get(key, 0.0)
        delta = s_val - p_val
        rows.append([key, fmt(s_val), fmt(p_val), fmt(delta)])

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + rows))]
    line_fmt = " | ".join(f"{{:{w}}}" for w in col_widths)

    print("\n=== Shadow vs. Paper‑Trading Summary ===")
    print(line_fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for row in rows:
        print(line_fmt.format(*row))

    # -------------------------------------------------------------
    # 7️⃣  Exit code – 0 = success, >0 = something went wrong
    # -------------------------------------------------------------
    print("\n✅ Shadow run completed.")
    return 0


# -----------------------------------------------------------------
# Helper: read a log file where each line is a JSON object.
# -----------------------------------------------------------------
def process_log(path: Path) -> pd.DataFrame:
    """
    Parse a line‑delimited JSON log file into a pandas DataFrame.

    Expected fields (all optional – missing columns are ignored):
        - timestamp (ISO‑8601 string)
        - latency_seconds (float)
        - success (bool/int)
        - slippage_pips (float)
        - reject_reason (string)

    Returns:
        pandas.DataFrame with one row per log entry.
    """
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}")

    if not records:
        return pd.DataFrame()   # empty DataFrame

    df = pd.DataFrame.from_records(records)

    # Normalise column names (some code may emit camelCase)
    df.rename(columns=lambda c: c.lower().replace("-", "_"), inplace=True)

    # Ensure proper dtypes where possible
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "success" in df:
        df["success"] = df["success"].astype(bool)
    if "latency_seconds" in df:
        df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce")
    if "slippage_pips" in df:
        df["slippage_pips"] = pd.to_numeric(df["slippage_pips"], errors="coerce")

    return df


# -----------------------------------------------------------------
# Entry‑point guard
# -----------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
