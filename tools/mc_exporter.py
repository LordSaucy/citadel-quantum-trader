#!/usr/bin/env python3
import os, time, subprocess, json
from prometheus_client import start_http_server, Gauge

# Gauges
MC_DD_99_9 = Gauge("mc_drawdown_99_9pct", "Monte‑Carlo 99.9 % draw‑down")
MC_MEDIAN   = Gauge("mc_median_equity", "Median equity after simulation")
MC_PROB_LOSS = Gauge("mc_prob_loss", "Probability final equity < 1.0")

def run_mc():
    # Run the simulation with a modest number of paths (e.g., 5 000) and JSON output
    cmd = [
        "python", "/opt/citadel/tools/simulate_wr.py",
        "--paths", "5000",
        "--json"
    ]
    out = subprocess.check_output(cmd, env=os.environ).decode()
    data = json.loads(out)

    MC_DD_99_9.set(data["dd_99_9pct"])
    MC_MEDIAN.set(data["median_equity"])
    MC_PROB_LOSS.set(data["prob_loss"])

if __name__ == "__main__":
    start_http_server(int(os.getenv("EXPORTER_PORT", "8002")))
    while True:
        try:
            run_mc()
        except Exception as exc:
            print(f"[mc_exporter] error: {exc}")
        time.sleep(int(os.getenv("REFRESH_SECONDS", "3600")))
