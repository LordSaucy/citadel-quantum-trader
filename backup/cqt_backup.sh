#!/usr/bin/env bash
# =============================================================================
# cqt_backup.sh – Nightly backup & archiving for Citadel Quantum Trader (CQT)
#
# What it does (in order):
#   1️⃣  Take a snapshot of the PostgreSQL block‑storage volume.
#   2️⃣  Prune old snapshots (retain only $RETENTION daily snapshots).
#   3️⃣  Archive CQT log files and upload them to a DigitalOcean Space.
#   4️⃣  (Optional) Upload any new PostgreSQL WAL files to the same Space.
#
# Requirements on the backup‑svc droplet:
#   • doctl   – DigitalOcean CLI (authenticated with $DO_TOKEN)
#   • aws     – AWS CLI (configured for Spaces endpoint)
#   • jq      – JSON processor (used for snapshot pruning)
#   • gzip, tar, date, mktemp – standard Unix utilities
#
# Secrets are read from Docker secrets (mounted at /run/secrets/*) or from
# environment variables when the script is executed locally.
#
# Author : Citadel Quantum Trader Team
# Version: 2024‑11‑28
# =============================================================================

set -euo pipefail

# ----------------------------------------------------------------------
# 0️⃣  Helper: read a secret (Docker secret → env var fallback)
# ----------------------------------------------------------------------
read_secret() {
    local secret_name="$1"   # Docker‑secret file name (e.g. DO_TOKEN)
    local env_name="$2"      # Corresponding environment variable name
    local value

    if [[ -f "/run/secrets/${secret_name}" ]]; then
        value=$(<"/run/secrets/${secret_name}")
    else
        value="${!env_name:-}"
    fi
    printf '%s' "$value"
    return 0                     # <-- explicit return for SonarQube
}

# ----------------------------------------------------------------------
# 1️⃣  Configuration (override via env vars or Docker secrets)
# ----------------------------------------------------------------------
DO_TOKEN="$(read_secret DO_TOKEN DO_TOKEN)"
SPACES_KEY="$(read_secret SPACES_KEY SPACES_KEY)"
SPACES_SECRET="$(read_secret SPACES_SECRET SPACES_SECRET)"
POSTGRES_VOLUME_ID="$(read_secret POSTGRES_VOLUME_ID POSTGRES_VOLUME_ID)"

# ----------------------------------------------------------------------
# 2️⃣  Operational parameters (tweak as needed)
# ----------------------------------------------------------------------
RETENTION=7                                 # keep the N newest snapshots
SPACES_BUCKET="cqt-backups"                # must already exist
SPACES_ENDPOINT="nyc3.digitaloceanspaces.com"
LOG_DIR="/var/log"
LOG_GLOB="cqt*.log*"                       # pattern for CQT logs
TMPDIR="$(mktemp -d -t cqt_backup_XXXXXX)" # temporary working dir

# ----------------------------------------------------------------------
# 3️⃣  Helper: structured logging (writes to syslog & local file)
# ----------------------------------------------------------------------
log() {
    local level="$1"   # e.g. info, warn, err, debug
    local msg="$2"
    # Syslog entry (facility=user)
    logger -t cqt_backup -p "user.${level}" "$msg"
    # Local file (append)
    printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$msg" \
        >> "/var/log/cqt_backup.log"
    return 0                     # <-- explicit return for SonarQube
}

# ----------------------------------------------------------------------
# 4️⃣  Helper: fatal error (log + exit)
# ----------------------------------------------------------------------
die() {
    local errmsg="$1"
    log "err" "$errmsg"
    exit 1                       # explicit exit (function never returns)
}

# ----------------------------------------------------------------------
# 5️⃣  Verify required binaries & environment variables
# ----------------------------------------------------------------------
for cmd in doctl aws jq tar gzip; do
    command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' is not installed"
done

[[ -n "$DO_TOKEN" ]]        || die "DO_TOKEN not set"
[[ -n "$SPACES_KEY" ]]      || die "SPACES_KEY not set"
[[ -n "$SPACES_SECRET" ]]   || die "SPACES_SECRET not set"
[[ -n "$POSTGRES_VOLUME_ID" ]] || die "POSTGRES_VOLUME_ID not set"

# ----------------------------------------------------------------------
# 6️⃣  Authenticate doctl (once per session)
# ----------------------------------------------------------------------
export DO_TOKEN
doctl auth init -t "$DO_TOKEN" >/dev/null 2>&1

# ----------------------------------------------------------------------
# 7️⃣  Create a snapshot of the PostgreSQL volume
# ----------------------------------------------------------------------
SNAP_NAME="cqt-pg-$(date +%Y%m%d-%H%M%S)"
log "info" "Creating snapshot '$SNAP_NAME' for volume $POSTGRES_VOLUME_ID"

SNAP_ID=$(doctl compute volume-action snapshot "$POSTGRES_VOLUME_ID" \
    --snapshot-name "$SNAP_NAME" \
    --format ID --no-header)

[[ -n "$SNAP_ID" ]] || die "Failed to create snapshot (empty ID returned)"
log "info" "Snapshot created: ID=$SNAP_ID, name=$SNAP_NAME"

# ----------------------------------------------------------------------
# 8️⃣  Prune old snapshots (keep only $RETENTION newest)
# ----------------------------------------------------------------------
log "info" "Pruning snapshots – retaining the latest $RETENTION"

# List snapshots for this volume, newest → oldest
SNAP_LIST=$(doctl compute snapshot list \
    --resource-type volume \
    --format ID,Name,CreatedAt \
    --no-header | \
    grep "$POSTGRES_VOLUME_ID" | \
    sort -k3 -r)

kept=0
while IFS= read -r line; do
    snap_id=$(awk '{print $1}' <<<"$line")
    snap_name=$(awk '{print $2}' <<<"$line")
    ((kept++))
    if (( kept > RETENTION )); then
        log "info" "Deleting old snapshot $snap_name (ID=$snap_id)"
        doctl compute snapshot delete "$snap_id" --force >/dev/null 2>&1 \
            || log "warn" "Failed to delete snapshot $snap_id (may have already been removed)"
    fi
done <<<"$SNAP_LIST"

log "info" "Snapshot pruning complete."

# ----------------------------------------------------------------------
# 9️⃣  Archive CQT log files and upload to Spaces
# ----------------------------------------------------------------------
log "info" "Archiving CQT logs from $LOG_DIR"

ARCHIVE_NAME="cqt-logs-$(date +%Y%m%d-%H%M%S).tar.gz"
ARCHIVE_PATH="${TMPDIR}/${ARCHIVE_NAME}"

# Build the tar.gz archive (ignore missing files)
if ! tar -czf "$ARCHIVE_PATH" -C "$LOG_DIR" $(ls $LOG_GLOB 2>/dev/null || true); then
    log "warn" "No log files matched pattern $LOG_GLOB – skipping upload"
else
    log "info" "Uploading log archive to Spaces bucket $SPACES_BUCKET"
    AWS_ACCESS_KEY_ID="$SPACES_KEY" \
    AWS_SECRET_ACCESS_KEY="$SPACES_SECRET" \
    aws s3 cp "$ARCHIVE_PATH" "s3://${SPACES_BUCKET}/${ARCHIVE_NAME}" \
        --endpoint-url "https://${SPACES_ENDPOINT}" \
        --storage-class STANDARD_IA >/dev/null 2>&1 && \
        log "info" "Log archive uploaded successfully" || \
        log "err" "Failed to upload log archive to Spaces"
fi

# ----------------------------------------------------------------------
# 🔟  (Optional) Archive new PostgreSQL WAL files
# ----------------------------------------------------------------------
# Adjust WAL_DIR if you mount the PostgreSQL volume elsewhere.
WAL_DIR="/mnt/pg_wal"

if [[ -d "$WAL_DIR" ]]; then
    log "info" "Scanning for new WAL files in $WAL_DIR"
    for wal_file in "$WAL_DIR"/*; do
        [[ -f "$wal_file" ]] || continue               # skip non‑regular files
        wal_name=$(basename "$wal_file")

        # Skip if already present in the bucket
        if AWS_ACCESS_KEY_ID="$SPACES_KEY" \
           AWS_SECRET_ACCESS_KEY="$SPACES_SECRET" \
           aws s3 ls "s3://${SPACES_BUCKET}/wal/${wal_name}" \
               --endpoint-url "https://${SPACES_ENDPOINT}" \
               >/dev/null 2>&1; then
            log "debug" "WAL $wal_name already present – skipping"
            continue
        fi

        log "info" "Uploading WAL $wal_name to bucket"
        AWS_ACCESS_KEY_ID="$SPACES_KEY" \
        AWS_SECRET_ACCESS_KEY="$SPACES_SECRET" \
        aws s3 cp "$wal_file" "s3://${SPACES_BUCKET}/wal/${wal_name}" \
            --endpoint-url "https://${SPACES_ENDPOINT}" \
            --storage-class STANDARD_IA >/dev/null 2>&1 && \
            log "info" "Uploaded $wal_name" || \
            log "err" "Failed to upload $wal_name"
    done
else
    log "warn" "WAL directory $WAL_DIR not found – skipping WAL archiving"
fi

# ----------------------------------------------------------------------
# 1️⃣1️⃣  Cleanup temporary workspace
# ----------------------------------------------------------------------
log "info" "Removing temporary directory $TMPDIR"
rm -rf "$TMPDIR"

log "info" "cqt_backup.sh completed successfully"
exit 0   # <-- explicit exit for SonarQube (function already ended)
