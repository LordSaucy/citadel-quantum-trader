#!/usr/bin/env bash
# /opt/citadel/scripts/run_collector.sh
set -euo pipefail

# Activate the virtualenv if you use one (skip if you run inside Docker)
# source /opt/citadel/venv/bin/activate

# Run the collector – the script is idempotent, so running it every hour is safe
python3 /opt/citadel/src/data_ingest/collector.py >> /opt/citadel/logs/collector.log 2>&1
