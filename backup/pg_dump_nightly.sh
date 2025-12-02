#!/usr/bin/env bash
set -euo pipefail

# ----- Configuration -------------------------------------------------
S3_BUCKET="citadel-audit"
DB_CONTAINER="citadel-db"
DB_NAME="citadel"
DB_USER="citadel"
# Pull the DB password from Vault (you can also read from an env var)
VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-$(cat /run/secrets/vault_token)}"
DB_PASSWORD=$(vault kv get -field=password secret/citadel/db)

# Timestamp for naming
TS=$(date -u +"%Y%m%dT%H%M%SZ")
DUMP_FILE="/tmp/citadel_dump_${TS}.sql.gz"

# ----- Run pg_dump inside the container -------------------------------
docker exec -i "${DB_CONTAINER}" pg_dump \
    -U "${DB_USER}" -d "${DB_NAME}" -Fc \
    -Z 9 -f "${DUMP_FILE}" \
    -v --no-owner --no-acl \
    --username="${DB_USER}" \
    --password="${DB_PASSWORD}" \
    --no-sync

# ----- Upload to S3 (versioned) --------------------------------------
aws s3 cp "${DUMP_FILE}" "s3://${S3_BUCKET}/db-dumps/citadel_dump_${TS}.sql.gz" \
    --storage-class STANDARD_IA \
    --sse aws:kms \
    --sse-kms-key-id "$(aws kms list-aliases --query "Aliases[?AliasName=='alias/citadel-backup'].TargetKeyId" --output text)"

# ----- Cleanup --------------------------------------------------------
rm -f "${DUMP_FILE}"
echo "✅ Nightly dump uploaded: citadel_dump_${TS}.sql.gz"
