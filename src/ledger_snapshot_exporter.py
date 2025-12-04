#!/usr/bin/env python3
import os
import time
import subprocess
from prometheus_client import start_http_server, Gauge

# Metric: 1 = last snapshot succeeded, 0 = failed
snapshot_success = Gauge(
    "ledger_snapshot_success",
    "1 if the most‑recent hourly ledger snapshot succeeded, 0 otherwise",
)

def run_snapshot() -> bool:
    """
    Calls the same function that the cron container uses.
    Returns True on success, False on any exception.
    """
    try:
        # The same entry‑point we used in the cron container
        subprocess.check_call(
            [
                "python",
                "-m",
                "src.trade_logger",
                "push_root_to_s3",
            ],
            cwd="/app",               # adjust if you mount elsewhere
            env=os.environ,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ledger snapshot failed: {e}")
        return False

def main():
    # Expose metrics on port 9200 (any free port works)
    start_http_server(9200)
    while True:
        ok = run_snapshot()
        snapshot_success.set(1 if ok else 0)
        # Sleep until the next hour boundary (e.g., 55 min to avoid overlap)
        time.sleep(55 * 60)

if __name__ == "__main__":
    main()
