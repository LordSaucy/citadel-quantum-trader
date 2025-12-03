#!/usr/bin/env python3
"""
Pushes the backup duration (seconds) to a Prometheus Pushgateway.
"""

import os
import sys
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def push(duration_seconds: float):
    registry = CollectorRegistry()
    g = Gauge(
        "cqt_backup_duration_seconds",
        "Duration of the nightly pg_dump → S3 upload",
        registry=registry,
    )
    g.set(duration_seconds)

    # Pushgateway address – set via env var or default to localhost:9091
    gateway = os.getenv("PROM_PUSHGATEWAY", "localhost:9091")
    job = os.getenv("PROM_BACKUP_JOB", "cqt_nightly_backup")
    try:
        push_to_gateway(gateway, job=job, registry=registry)
        print(f"[backup‑metric] Pushed {duration_seconds:.1f}s to {gateway}/{job}")
    except Exception as e:
        print(f"[backup‑metric] FAILED to push metric: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: push_backup_metric.py <seconds>", file=sys.stderr)
        sys.exit(1)
    push(float(sys.argv[1]))
