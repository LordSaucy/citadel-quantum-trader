#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------
# Variables – adjust if you change the volume mount point
# -----------------------------------------------------------------
LOKI_DIR="/var/lib/loki"
S3_BUCKET="s3://citadel-logs/loki"
DATE_SUFFIX="$(date +%Y-%m-%d)"
DEST="${S3_BUCKET}/${DATE_SUFFIX}"

# -----------------------------------------------------------------
# Sync – only new/changed files are transferred (fast)
# -----------------------------------------------------------------
aws s3 sync "${LOKI_DIR}" "${DEST}" \
    --storage-class GLACIER \
    --only-show-errors

# Optional: keep a local manifest for audit purposes
echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") exported ${LOKI_DIR} → ${DEST}" >> /var/log/loki_export.log
