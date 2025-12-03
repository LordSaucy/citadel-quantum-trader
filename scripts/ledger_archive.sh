# scripts/ledger_archive.sh
#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------
# Configuration – adjust to your environment
# -----------------------------------------------------------------
LEDGER_DIR="/opt/citadel/ledger"          # where the bot writes snapshots
S3_BUCKET="citadel-audit"
S3_PREFIX="paper_runs/$(date +%Y%m%d_%H%M%S)"   # unique folder per run
AWS_PROFILE="${AWS_PROFILE:-default}"      # if you use a named profile
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 1️⃣ Sync the whole ledger directory (preserves all snapshots)
# -----------------------------------------------------------------
aws s3 sync "${LEDGER_DIR}/" "s3://${S3_BUCKET}/${S3_PREFIX}/" \
    --acl bucket-owner-full-control \
    --storage-class STANDARD_IA \
    --exclude "*" --include "ledger_snapshot_*.json"

# -----------------------------------------------------------------
# 2️⃣ (Optional) Also copy the final SQLite/PG dump if you keep one
# -----------------------------------------------------------------
if [[ -f "${LEDGER_DIR}/ledger.db" ]]; then
    aws s3 cp "${LEDGER_DIR}/ledger.db" "s3://${S3_BUCKET}/${S3_PREFIX}/ledger.db"
fi

echo "✅ Ledger archive uploaded to s3://${S3_BUCKET}/${S3_PREFIX}/"
