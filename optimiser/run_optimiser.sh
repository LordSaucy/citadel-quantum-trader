# /opt/citadel/scripts/run_optimiser.sh
#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="/opt/citadel/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%F_%H%M)
LOG_FILE="${LOG_DIR}/optimiser_${DATE}.log"


echo "=== Optimiser start $(date) ===" | tee -a "$LOG_FILE"
cd /opt/citadel
