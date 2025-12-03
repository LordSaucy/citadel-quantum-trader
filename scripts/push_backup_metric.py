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

#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------
# Existing backup logic (pg_dump → S3) – keep unchanged
# -----------------------------------------------------------------
START_TS=$(date +%s)

# Example: run pg_dump and upload to S3
docker exec citadel-db pg_dump -U citadel citadel > /tmp/backup.sql
aws s3 cp /tmp/backup.sql s3://citadel-audit/backup/backup_$(date +%Y%m%d_%H%M).sql

END_TS=$(date +%s)
DURATION=$(( END_TS - START_TS ))

echo "✅ Backup completed in ${DURATION}s"

# -----------------------------------------------------------------
# NEW: push the duration metric to Prometheus Pushgateway
# -----------------------------------------------------------------
# The Pushgateway should be reachable from the host (e.g., running on
# the same VPS on port 9091). If you run it inside Docker, expose it:
#   docker run -d -p 9091:9091 prom/pushgateway
#
# Push the metric (ignore failures – we still want the backup to succeed)
python3 /opt/citadel/scripts/push_backup_metric.py "${DURATION}" || true

