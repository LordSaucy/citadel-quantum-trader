from prometheus_client import start_http_server, Gauge
import time, subprocess, json, os

DD_GAUGE = Gauge("mc_drawdown_99_9pct", "Monte‑Carlo 99.9 % draw‑down")
WINRATE_GAUGE = Gauge("mc_winrate_estimate", "Monte‑Carlo win‑rate estimate")

def run_mc():
    # call the script and parse the printed line that contains the 99.9 % DD
    result = subprocess.check_output(["python", "simulate_wr.py", "--paths", "2000", "--plot", "0"])
    # simple regex extraction (you can make the script output JSON if you prefer)
    for line in result.decode().splitlines():
        if line.startswith("5‑th / 95‑th percentile"):
            # line format: "5‑th / 95‑th percentile : 2.3000× – 2.6000×"
            continue
        if "99.9 %" in line.lower():
            # Example: "Monte‑Carlo 99.9 % DD = 0.048"
            dd = float(line.split("=")[1].strip().replace("%",""))
            DD_GAUGE.set(dd/100)
        if "win‑rate (input)" in line.lower():
            wr = float(line.split(":")[1].strip().replace("%",""))/100
            WINRATE_GAUGE.set(wr)

if __name__ == "__main__":
    start_http_server(8002)   # expose on localhost:8002/metrics
    while True:
        run_mc()
        time.sleep(3600)      # refresh hourly
