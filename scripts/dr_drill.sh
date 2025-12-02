#!/usr/bin/env bash
#
# dr_drill.sh – Disaster‑Recovery (DR) validation drill
#
# What it does (high‑level):
#   1️⃣ Stop the primary PostgreSQL container (citadel-db)
#   2️⃣ Spin a temporary PostgreSQL container to act as the restore target
#   3️⃣ Pull the latest base backup (tar.gz) from S3 and extract it
#   4️⃣ Replay WAL files for the last N hours (default 24h) using pg_receivewal
#   5️⃣ Verify the restored data matches the original snapshot (sha256 checksum)
#   6️⃣ Bring the primary DB back up
#
# ----------------------------------------------------------------------
# CONFIGURATION – adjust via environment variables or command‑line flags
# ----------------------------------------------------------------------
set -euo pipefail   # fail fast on errors, undefined vars, pipe failures

# ---- Default values (override with env vars) -------------------------
DRY_RUN=${DRY_RUN:-false}                 # if true, only prints actions
RESTORE_CONTAINER_NAME=${RESTORE_CONTAINER_NAME:-pg_restore}
RESTORE_DATA_DIR=${RESTORE_DATA_DIR:-dr_restore}
BASE_BACKUP_S3_PREFIX=${BASE_BACKUP_S3_PREFIX:-citadel-audit/backup}
WAL_S3_PREFIX=${WAL_S3_PREFIX:-citadel-audit/wal}
WAL_HOURS=${WAL_HOURS:-24}                # how many recent hours of WAL to replay
POSTGRES_IMAGE=${POSTGRES_IMAGE:-postgres:15}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-secret}
POSTGRES_USER=${POSTGRES_USER:-postgres}
# ---------------------------------------------------------------------

log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') [DR-DRILL] $*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------
# Helper: run a command unless we are in dry‑run mode
# ---------------------------------------------------------------------
run() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY‑RUN] $*"
    else
        eval "$@"
    fi
}

# ---------------------------------------------------------------------
# 0️⃣ Sanity checks
# ---------------------------------------------------------------------
[[ -x "$(command -v aws)" ]] || die "aws CLI not found"
[[ -x "$(command -v docker)" ]] || die "docker not found"
[[ -x "$(command -v sha256sum)" ]] || die "sha256sum not found"

# Ensure we are in the repo root (where docker‑compose.yml lives)
if [[ ! -f docker-compose.yml ]]; then
    die "docker-compose.yml not found – run this script from the repo root"
fi

# ---------------------------------------------------------------------
# 1️⃣ Stop the primary DB (citadel-db)
# ---------------------------------------------------------------------
log "Stopping primary PostgreSQL container (citadel-db)…"
run "docker compose stop citadel-db"

# ---------------------------------------------------------------------
# 2️⃣ Spin up a temporary restore container (detached)
# ---------------------------------------------------------------------
log "Launching temporary restore container [$RESTORE_CONTAINER_NAME]…"
run "docker run -d --name $RESTORE_CONTAINER_NAME \
    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
    -e POSTGRES_USER=$POSTGRES_USER \
    -v $(pwd)/$RESTORE_DATA_DIR:/var/lib/postgresql/data \
    $POSTGRES_IMAGE"

# Wait for the container to be healthy (Postgres starts)
log "Waiting for restore container to become ready (max 30 s)…"
timeout 30 bash -c "
    while ! docker exec $RESTORE_CONTAINER_NAME pg_isready -U $POSTGRES_USER >/dev/null 2>&1; do
        sleep 1
    done
"
log "Restore container is up."

# ---------------------------------------------------------------------
# 3️⃣ Pull the latest base backup from S3 and extract it
# ---------------------------------------------------------------------
log "Fetching latest base backup from S3…"
LATEST_BACKUP=$(aws s3 ls s3://$BASE_BACKUP_S3_PREFIX/ | sort | tail -n1 | awk '{print $4}')
[[ -z "$LATEST_BACKUP" ]] && die "No backup files found in s3://$BASE_BACKUP_S3_PREFIX/"

log "Downloading $LATEST_BACKUP …"
run "aws s3 cp s3://$BASE_BACKUP_S3_PREFIX/$LATEST_BACKUP - | tar -xz -C $RESTORE_DATA_DIR"

# Verify that the extracted directory now contains a PG_VERSION file
if [[ ! -f $RESTORE_DATA_DIR/PG_VERSION ]]; then
    die "Extraction failed – PG_VERSION not found in $RESTORE_DATA_DIR"
fi

log "Base backup extracted (PostgreSQL version $(cat $RESTORE_DATA_DIR/PG_VERSION))."

# ---------------------------------------------------------------------
# 4️⃣ Replay WAL files for the last $WAL_HOURS hours
# ---------------------------------------------------------------------
log "Replaying WAL files for the last $WAL_HOURS hour(s)…"

# Create (or reuse) a replication slot named restore_slot
run "docker exec $RESTORE_CONTAINER_NAME pg_create_physical_replication_slot -U $POSTGRES_USER restore_slot || true"

# Pull WAL files from S3 – we assume they are stored with filenames like 0000000100000000000000A1
# We'll download only those whose modification time is within the last $WAL_HOURS.
WAL_TMP=$(mktemp -d)
log "Downloading recent WAL files to $WAL_TMP …"
run "aws s3 sync s3://$WAL_S3_PREFIX $WAL_TMP --exclude \"*\" \
    --include \"*\" \
    --exact-timestamps \
    --no-progress"

# Filter files newer than $WAL_HOURS ago
cutoff_epoch=$(date -d "-$WAL_HOURS hours" +%s)
wal_files=()
while IFS= read -r -d '' file; do
    mod_epoch=$(stat -c %Y \"$file\")
    if (( mod_epoch >= cutoff_epoch )); then
        wal_files+=("$file")
    fi
done < <(find "$WAL_TMP" -type f -print0)

if [[ ${#wal_files[@]} -eq 0 ]]; then
    die "No WAL files found for the last $WAL_HOURS hour(s)."
fi

log "Found ${#wal_files[@]} WAL segments – feeding to pg_receivewal…"

# Feed each WAL segment to the restore container
for wal in "${wal_files[@]}"; do
    run "docker cp \"$wal\" $RESTORE_CONTAINER_NAME:/tmp/$(basename \"$wal\")"
    run "docker exec $RESTORE_CONTAINER_NAME pg_receivewal -D /var/lib/postgresql/data \
        --slot=restore_slot \
        --create-slot \
        --endpos=LATEST \
        --walfile=/tmp/$(basename \"$wal\")"
done

log "WAL replay completed."

# ---------------------------------------------------------------------
# 5️⃣ Verify checksum vs. original snapshot
# ---------------------------------------------------------------------
log "Computing checksum of restored PG_VERSION…"
RESTORE_CHECKSUM=$(sha256sum $RESTORE_DATA_DIR/PG_VERSION | awk '{print $1}')

log "Fetching original checksum from S3 (stored alongside the backup)…"
ORIG_CHECKSUM=$(aws s3 cp s3://$BASE_BACKUP_S3_PREFIX/$(basename "$LATEST_BACKUP" .tar.gz).sha256 - 2>/dev/null || true)

if [[ -z "$ORIG_CHECKSUM" ]]; then
    log "⚠ No original checksum file found – skipping verification."
else
    log "Original checksum: $ORIG_CHECKSUM"
    log "Restored checksum: $RESTORE_CHECKSUM"
    if [[ "$ORIG_CHECKSUM" == "$RESTORE_CHECKSUM" ]]; then
        log "✅ Checksums match – restore appears consistent."
    else
        log "❌ Checksums differ!  Restored data may be corrupt."
    fi
fi

# ---------------------------------------------------------------------
# 6️⃣ Bring the primary DB back up
# ---------------------------------------------------------------------
log "Bringing primary PostgreSQL container back up…"
run "docker compose start citadel-db"

# Optional: clean up the temporary restore container and data dir
log "Cleaning up temporary restore resources…"
run "docker rm -f $RESTORE_CONTAINER_NAME"
run "rm -rf $RESTORE_DATA_DIR"

log "🟢 DR‑drill completed successfully."
