#!/usr/bin/env python3
"""
run_shadow.py

Launches the Citadel stack in SHADOW mode, runs for a configurable
duration (or until a trade count is reached), stops the stack,
and then compares the shadow log against the paper‑trading log.

Shadow mode: Engine trades without real capital, mirrors live signals.
Used for validation before paper/live trading.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

# -----------------------------------------------------------------
# Configuration (adjust as needed)
# -----------------------------------------------------------------
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
BOT_HEALTH_URL = "http://localhost:8000/health"
METRIC_POLL_INTERVAL = 30          # seconds
MAX_TRADE_COUNT = 500
MAX_DURATION = timedelta(hours=48)  # 48 h


def wait_for_bot(timeout: int = 120) -> bool:
    """Wait for the bot health endpoint to respond successfully."""
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
    """Query Prometheus and return a single float value."""
    resp = requests.get(PROMETHEUS_URL, params={"query": expr})
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success" or not data["data"]["result"]:
        return 0.0
    return float(data["data"]["result"][0]["value"][1])


# =========================================================================
# Monitoring loop
# =========================================================================
def run_monitoring_loop() -> Dict[str, Any]:
    """
    Monitor metrics in a loop until MAX_TRADE_COUNT or MAX_DURATION is reached.
    
    Returns a dict with collected metrics.
    """
    start_time = datetime.now()
    trade_counter = 0
    max_latency = 0.0

    while True:
        latency = prom_query('max(cqt_order_latency_seconds{shadow="yes"})')
        max_latency = max(max_latency, latency)

        trade_counter = int(prom_query('sum(cqt_orders_total{shadow="yes"})'))

        elapsed = datetime.now() - start_time
        if trade_counter >= MAX_TRADE_COUNT:
            print(f"🏁 Reached {trade_counter} shadow trades – stopping")
            break
        if elapsed >= MAX_DURATION:
            print(f"⌛ 48 h elapsed ({elapsed}) – stopping")
            break

        time.sleep(METRIC_POLL_INTERVAL)

    return {
        "trade_counter": trade_counter,
        "max_latency": max_latency,
        "elapsed": elapsed,
    }


# =========================================================================
# Docker stack operations
# =========================================================================
def start_docker_stack(compose_file: str, override_file: str) -> bool:
    """Start the Docker stack in shadow mode."""
    print("🚀 Starting Shadow stack …")
    up_cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "-f",
        override_file,
        "up",
        "-d",
    ]
    try:
        subprocess.check_call(up_cmd)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to start Docker stack: {exc}", file=sys.stderr)
        return False


def stop_docker_stack(compose_file: str, override_file: str) -> None:
    """Stop the Docker stack."""
    print("🛑 Stopping Docker stack …")
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "-f", override_file, "down"],
        check=False,
    )


# =========================================================================
# ✅ FIXED: Removed unused parameters 'compose_file' and 'override_file'
#           They are not used in this function's logic
# =========================================================================
def retrieve_logs() -> tuple[bool, Optional[Path], Optional[Path]]:
    """
    Retrieve shadow and paper logs from the container.
    
    Assumes the cqt-engine container has already been stopped or is
    available for `docker cp` operations.
    
    Returns:
        Tuple of (success, shadow_log_path, paper_log_path)
        - success: bool indicating if both logs were retrieved
        - shadow_log_path: Path to shadow log (or None if retrieval failed)
        - paper_log_path: Path to paper log (or None if retrieval failed)
    """
    print("📂 Retrieving logs …")
    shadow_log_host_path = Path("/tmp/cqt_shadow.log")
    paper_log_host_path = Path("/tmp/cqt_paper.log")

    try:
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
        return True, shadow_log_host_path, paper_log_host_path
    except subprocess.CalledProcessError as exc:
        print(f"⚠️  Could not retrieve logs: {exc}", file=sys.stderr)
        return False, None, None


# =========================================================================
# ✅ FIXED: Reduced cognitive complexity from 16 to 14 by extracting
#           a helper function for metric summarization
# =========================================================================
def _summarize_log_metrics(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """
    Extract key performance metrics from a log DataFrame.
    
    Args:
        df: DataFrame with log entries
        label: Label for this log (e.g., "SHADOW" or "PAPER")
    
    Returns:
        Dict with aggregated metrics
    """
    out: Dict[str, Any] = {}
    
    if "latency_seconds" in df.columns:
        out["avg_latency"] = df["latency_seconds"].mean()
        out["max_latency"] = df["latency_seconds"].max()
    
    if "success" in df.columns:
        successes = df["success"].astype(bool).sum()
        total = len(df)
        out["win_rate"] = successes / total * 100 if total else 0.0
    
    if "slippage_pips" in df.columns:
        out["avg_slip"] = df["slippage_pips"].mean()
    
    if "reject_reason" in df.columns:
        out["rejects"] = df["reject_reason"].notna().sum()
    
    out["label"] = label
    return out


def _format_value(v: Any) -> str:
    """Format a value for display (float or string)."""
    return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)


def compare_logs(shadow_df: pd.DataFrame, paper_df: pd.DataFrame) -> int:
    """
    Compare shadow and paper logs, print results.
    
    ✅ FIXED: Extracted summarization logic to reduce complexity from 16 to 14
    
    Args:
        shadow_df: DataFrame with shadow trades
        paper_df: DataFrame with paper trades
    
    Returns:
        Exit code (0 = success, 4 = incomparable logs)
    """
    print("🔎 Analysing logs …")
    
    # Basic sanity check – both logs should contain some common columns
    common_cols = set(shadow_df.columns) & set(paper_df.columns)
    if not common_cols:
        print("⚠️  No overlapping columns between logs – cannot compare.", file=sys.stderr)
        return 4

    # Summarize metrics from both logs
    shadow_stats = _summarize_log_metrics(shadow_df, "SHADOW")
    paper_stats = _summarize_log_metrics(paper_df, "PAPER ")

    # Build comparison table
    headers = ["Metric", "Shadow", "Paper", "Δ (Shadow‑Paper)"]
    rows = []
    metric_keys = set(shadow_stats) | set(paper_stats)
    metric_keys.discard("label")

    for key in sorted(metric_keys):
        s_val = shadow_stats.get(key, 0.0)
        p_val = paper_stats.get(key, 0.0)
        delta = s_val - p_val
        rows.append([key, _format_value(s_val), _format_value(p_val), _format_value(delta)])

    # Format and print table
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + rows))]
    line_fmt = " | ".join(f"{{:{w}}}" for w in col_widths)

    print("\n=== Shadow vs. Paper‑Trading Summary ===")
    print(line_fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for row in rows:
        print(line_fmt.format(*row))

    return 0


def main() -> int:
    """
    Main entry point – orchestrate the shadow run.
    
    Workflow:
    1️⃣ Start Docker stack
    2️⃣ Wait for bot health
    3️⃣ Run monitoring loop
    4️⃣ Stop Docker stack
    5️⃣ Retrieve logs
    6️⃣ Load and compare logs
    7️⃣ Return results
    """
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

    # 1️⃣ Start Docker stack
    if not start_docker_stack(args.compose, args.override):
        return 1

    # 2️⃣ Wait for health endpoint
    print("⏳ Waiting for bot health …")
    if not wait_for_bot():
        print("❌ Bot never became healthy – aborting", file=sys.stderr)
        stop_docker_stack(args.compose, args.override)
        return 1

    print("✅ Bot is healthy – monitoring metrics …")

    # 3️⃣ Run monitoring loop
    # ✅ FIXED: Removed unused assignment of 'metrics'
    run_monitoring_loop()

    # 4️⃣ Stop Docker stack
    stop_docker_stack(args.compose, args.override)

    # 5️⃣ Retrieve logs
    success, shadow_path, paper_path = retrieve_logs()
    if not success:
        return 2

    # Guard against None values
    if shadow_path is None or paper_path is None:
        print("❌ Failed to retrieve log paths", file=sys.stderr)
        return 2

    # 6️⃣ Load and compare logs
    print("📂 Loading logs …")
    try:
        shadow_df = process_log(shadow_path)
        paper_df = process_log(paper_path)
    except Exception as exc:
        print(f"❌ Failed to parse logs: {exc}", file=sys.stderr)
        return 3

    # 7️⃣ Compare and print results
    result = compare_logs(shadow_df, paper_df)
    if result == 0:
        print("\n✅ Shadow run completed.")

    return result


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

    Args:
        path: Path to the log file

    Returns:
        pandas.DataFrame with one row per log entry.
    
    Raises:
        ValueError: If JSON parsing fails
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
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc

    if not records:
        return pd.DataFrame()   # empty DataFrame

    df = pd.DataFrame.from_records(records)

    # Normalise column names (some code may emit camelCase)
    df.rename(columns=lambda c: c.lower().replace("-", "_"), inplace=True)

    # Ensure proper dtypes where possible
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "success" in df.columns:
        df["success"] = df["success"].astype(bool)
    if "latency_seconds" in df.columns:
        df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce")
    if "slippage_pips" in df.columns:
        df["slippage_pips"] = pd.to_numeric(df["slippage_pips"], errors="coerce")

    return df


# -----------------------------------------------------------------
# Entry‑point guard
# -----------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
