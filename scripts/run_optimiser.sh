#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="/opt/citadel/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%F_%H%M)
LOG_FILE="${LOG_DIR}/optimiser_${DATE}.log"


echo "=== Optimiser start $(date) ===" | tee -a "$LOG_FILE"
cd /opt/citadel
# Pull the latest image (optional, if you push new versions)
docker compose pull optimiser || true
# Run the optimiser container; forward its stdout/stderr to the log file
docker compose run --rm optimiser >>"$LOG_FILE" 2>&1
echo "=== Optimiser end $(date) ===" | tee -a "$LOG_FILE"
