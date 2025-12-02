#!/usr/bin/env bash
set -euo pipefail

# Variables – adjust to your bucket / region
BUCKET="s3://my-citadel-loki-backups"
LOCAL_DIR="/opt/citadel/loki-backup"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%SZ")

# Ensure the local dir exists
mkdir -p "${LOCAL_DIR}"

# 1️⃣ Dump the Loki data directory (the Docker volume) to a tarball
docker run --rm \
  -v citadel-loki-data:/loki \
  -v "${LOCAL_DIR}:/backup" \
  alpine \
  tar czf "/backup/loki_${TIMESTAMP}.tar.gz" -C /loki .

# 2️⃣ Sync to S3 (requires AWS CLI configured with a role that can write to the bucket)
aws s3 cp "${LOCAL_DIR}/loki_${TIMESTAMP}.tar.gz" "${BUCKET}/"

# 3️⃣ (Optional) Delete local tarballs older than 7 days to keep the host tidy
find "${LOCAL_DIR}" -type f -name "loki_*.tar.gz" -mtime +7 -delete
