#!/usr/bin/env python3
import os
import time
from prometheus_client import start_http_server, Gauge

# Path where the dump script writes its status line
STATUS_FILE = "/opt/backup/pg_dump_status.txt"

# Two gauges – one for the *timestamp* of the last successful dump,
# another for a *boolean* success flag (1 = ok, 0 = failure)
last_success_ts = Gauge('citadel_pg_dump_last_success_timestamp',
                       'Unix timestamp of the most recent successful pg_dump')
last_success_ok = Gauge('citadel_pg_dump_last_success',
                       '1 if the most recent dump succeeded, 0 otherwise')

def read_status():
    try:
        with open(STATUS_FILE, "r") as f:
            line = f.read().strip()
        # Expected format: "SUCCESS 2024-09-15T02:30:00Z"
        status, ts = line.split()
        ts_epoch = int(time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")))
        return status, ts_epoch
    except Exception:
        return "UNKNOWN", 0

if __name__ == "__main__":
    start_http_server(9110)   # expose on port 9110
    while True:
        status, ts = read_status()
        if status == "SUCCESS":
            last_success_ok.set(1)
            last_success_ts.set(ts)
        else:
            last_success_ok.set(0)
        time.sleep(30)
