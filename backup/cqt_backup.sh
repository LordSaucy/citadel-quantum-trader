#!/usr/bin/env bash
# ----------------------------------------------------------------------
# cqt_backup.sh – Nightly backup & archiving for Citadel Quantum Trader
#
# What it does:
#   1️⃣  Takes a snapshot of the PostgreSQL block‑storage volume.
#   2️⃣  Prunes old snapshots (retain only $RETENTION daily snapshots).
#   3️⃣  Archives CQT log files and streams them to a DigitalOcean Space.
#   4️⃣  (Optional) Copies any new PostgreSQL WAL files to the same Space.
#
# Requirements (on the backup‑svc droplet):
#   * doctl   – DigitalOcean CLI (authenticated with $DO_TOKEN)
#   * aws     – AWS CLI (configured for Spaces endpoint)
#   * jq      – JSON processor (used for snapshot pruning)
#   * gzip, tar, date, mktemp – standard Unix utilities
#
# All secrets are read from Docker secrets (mounted at /run/secrets/*) or
# from environment variables if the script is run locally.
#
# Author:  Citadel Quantum Trader Team
# Version: 2024‑11‑28
# ----------------------------------------------------------------------

set -euo pipefail

# ----------------------------------------------------------------------
# 0️⃣  Configuration (override via env vars or Docker secrets)
# ----------------------------------------------------------------------
# ----- Secrets (Docker‑secret names) ------------------------------------
# The script will look for these files under /run/secrets/.  If they do
# not exist it falls back to the corresponding environment variable.
# ----------------------------------------------------------------------
read_secret() {
    local name="$1"
    local var_name="$2"
    if [[ -f "/run/secrets/${name}" ]]; then
        cat "/run/secrets/${name}"
    else
        printf '%s' "${!var_name:-}"
    fi
}

# DigitalOcean API token (required for doctl)
DO_TOKEN="$(read_secret DO_TOKEN DO_TOKEN)"

# Spaces credentials (required for aws cli)
SPACES_KEY="$(read_secret SPACES_KEY SPACES_KEY)"
SPACES_SECRET="$(read_secret SPACES_SECRET SPACES_SECRET)"

# PostgreSQL volume ID (the block‑storage volume that holds the DB data)
POSTGRES_VOLUME_ID="$(read_secret POSTGRES_VOLUME_ID POSTGRES_VOLUME_ID)"

# ----------------------------------------------------------------------
# ----- Operational parameters (feel free to tweak) ----------------------
# ----------------------------------------------------------------------
# Number of daily snapshots to retain (oldest snapshots beyond this are deleted)
RETENTION=7

# Spaces bucket name (must already exist in your DO account)
SPACES_BUCKET="cqt-backups"

# Spaces endpoint – change region if you store the bucket elsewhere
# Example for nyc3: nyc3.digitaloceanspaces.com
SPACES_ENDPOINT="nyc3.digitaloceanspaces.com"

# Log directory on the backup‑svc host (where CQT writes its logs)
LOG_DIR="/var/log"

# Sub‑directory pattern for CQT logs (adjust if you store logs elsewhere)
LOG_GLOB="cqt*.log*"

# Temporary working directory (will be removed automatically)
TMPDIR="$(mktemp -d -t cqt_backup_XXXXXX)"

# ----------------------------------------------------------------------
# 1️⃣  Helper functions
# ----------------------------------------------------------------------
log() {
    local level="$1"
    local msg="$2"
    # Write to syslog and to the local logfile
    logger -t cqt_backup -p "user.${level}" "$msg"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $msg" >> "/var/log/cqt_backup.log"
}

die() {
    log "err" "$1"
    exit 1
}

# ----------------------------------------------------------------------
# 2️⃣  Verify required tools & env
# ----------------------------------------------------------------------
command -v doctl >/dev/null 2>&1 || die "doctl not installed"
command -v aws >/dev/null 2>&1   || die "aws CLI not installed"
command -v jq >/dev/null 2>&1    || die "jq not installed"
command -v tar >/dev/null 2>&1   || die "tar not installed"
command -v gzip >/dev/null 2>&1 || die "gzip not installed"

[[ -n "$DO_TOKEN" ]]        || die "DO_TOKEN not set"
[[ -n "$SPACES_KEY" ]]      || die "SPACES_KEY not set"
[[ -n "$SPACES_SECRET" ]]   || die "SPACES_SECRET not set"
[[ -n "$POSTGRES_VOLUME_ID" ]] || die "POSTGRES_VOLUME_ID not set"

# ----------------------------------------------------------------------
# 3️⃣  Authenticate doctl (only needed once per session)
# ----------------------------------------------------------------------
export DO_TOKEN
doctl auth init -t "$DO_TOKEN" >/dev/null 2>&1

# ----------------------------------------------------------------------
# 4️⃣  Create a snapshot of the PostgreSQL volume
# ----------------------------------------------------------------------
SNAP_NAME="cqt-pg-$(date +%Y%m%d-%H%M%S)"
log "info" "Creating snapshot '$SNAP_NAME' for volume $POSTGRES_VOLUME_ID"

SNAP_ID=$(doctl compute volume-action snapshot "$POSTGRES_VOLUME_ID" \
    --snapshot-name "$SNAP_NAME" \
    --format ID --no-header)

if [[ -z "$SNAP_ID" ]]; then
    die "Failed to create snapshot (empty ID returned)"
fi
log "info" "Snapshot created: ID=$SNAP_ID, name=$SNAP_NAME"

# ----------------------------------------------------------------------
# 5️⃣  Prune old snapshots (keep only $RETENTION newest)
# ----------------------------------------------------------------------
log "info" "Pruning snapshots – retaining the latest $RETENTION"

# Get a list of snapshots for this volume, sorted newest→oldest
SNAP_LIST=$(doctl compute snapshot list \
    --resource-type volume \
    --format ID,Name,CreatedAt \
    --no-header | \
    grep "$POSTGRES_VOLUME_ID" | \
    sort -k3 -r)

# Counter for how many we have kept so far
kept=0

while IFS= read -r line; do
    snap_id=$(echo "$line" | awk '{print $1}')
    snap_name=$(echo "$line" | awk '{print $2}')
    ((kept++))
    if (( kept > RETENTION )); then
        log "info" "Deleting old snapshot $snap_name (ID=$snap_id)"
        doctl compute snapshot delete "$snap_id" --force >/dev/null 2>&1 || \
            log "warn" "Failed to delete snapshot $snap_id (may have already been removed)"
    fi
done <<< "$SNAP_LIST"

log "info" "Snapshot pruning complete."

# ----------------------------------------------------------------------
# 6️⃣  Archive CQT log files and upload to Spaces
# ----------------------------------------------------------------------
log "info" "Archiving CQT logs from $LOG_DIR"

ARCHIVE_NAME="cqt-logs-$(date +%Y%m%d-%H%M%S).tar.gz"
ARCHIVE_PATH="${TMPDIR}/${ARCHIVE_NAME}"

# Create the tar.gz archive (exclude anything that is not a CQT log)
tar -czf "$ARCHIVE_PATH" -C "$LOG_DIR" $(ls $LOG_GLOB 2>/dev/null || true)

if [[ ! -s "$ARCHIVE_PATH" ]]; then
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
# 7️⃣  (Optional) Archive new PostgreSQL WAL files
# ----------------------------------------------------------------------
# This step assumes you have WAL archiving enabled in postgresql.conf:
#   archive_mode = on
#   archive_command = 'aws --endpoint-url=https://nyc3.digitaloceanspaces.com s3 cp %p s3://cqt-wal-archive/%f'
# The command below simply mirrors that behaviour for any WAL files that
# have not yet been uploaded (it looks in the pg_wal directory inside the
# mounted volume).

WAL_DIR="/mnt/pg_wal"   # Adjust if you mount the volume elsewhere
if [[ -d "$WAL_DIR" ]]; then
    log "info" "Scanning for new WAL files in $WAL_DIR"
    for wal_file in "$WAL_DIR"/*; do
        # Skip if it's not a regular file
        [[ -f "$wal_file" ]] || continue

        # Derive the remote object name (just the filename)
        wal_name=$(basename "$wal_file")

        # Check if the file already exists in the bucket
        if AWS_ACCESS_KEY_ID="$SPACES_KEY" \
           AWS_SECRET_ACCESS_KEY="$SPACES_SECRET" \
           aws s3 ls "s3://${SPACES_BUCKET}/wal/${wal_name}" \
               --endpoint-url "https://${SPACES_ENDPOINT}" \
               >/dev/null 2>&1; then
            log "debug" "WAL $wal_name already present in bucket – skipping"
            continue
        fi

        # Upload the WAL file
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
# 8️⃣  Cleanup
# ----------------------------------------------------------------------
log "info" "Cleaning up temporary files"
rm -rf "$TMPDIR"

log "info" "cqt_backup.sh completed successfully"
exit 0
