#!/usr/bin/env bash
# -------------------------------------------------
# backup_verify.sh
# 1️⃣ Compute SHA‑256 of the latest pg_dump file
# 2️⃣ Store the checksum alongside the dump in S3
# -------------------------------------------------
set -euo pipefail

# ==== CONFIG ==============================================================
BUCKET="citadel-audit"
DUMP_PREFIX="backup/"                 # e.g. s3://citadel-audit/backup/
HASH_SUFFIX="${DUMP_PREFIX}hashes/"   # where we keep the .sha256 files
# ========================================================================

# 1️⃣ Find the *most recent* dump file (assumes .dump extension)
LATEST_DUMP=$(aws s3 ls "s3://${BUCKET}/${DUMP_PREFIX}" --recursive \
               | sort -k1,2 | tail -n1 | awk '{print $4}')

if [[ -z "${LATEST_DUMP}" ]]; then
  echo "❌ No dump found in s3://${BUCKET}/${DUMP_PREFIX}"
  exit 1
fi

echo "🔎 Latest dump: ${LATEST_DUMP}"

# 2️⃣ Download the dump locally (in /tmp)
TMP_DUMP="/tmp/$(basename "${LATEST_DUMP}")"
aws s3 cp "s3://${BUCKET}/${LATEST_DUMP}" "${TMP_DUMP}"

# 3️⃣ Compute SHA‑256
CHECKSUM=$(sha256sum "${TMP_DUMP}" | awk '{print $1}')
echo "✅ Checksum: ${CHECKSUM}"

# 4️⃣ Upload the checksum file (same base name, .sha256 suffix)
HASH_OBJ="${HASH_SUFFIX}$(basename "${LATEST_DUMP}").sha256"
printf "%s  %s\n" "${CHECKSUM}" "$(basename "${LATEST_DUMP}")" | \
    aws s3 cp - "s3://${BUCKET}/${HASH_OBJ}"

echo "💾 Uploaded checksum to s3://${BUCKET}/${HASH_OBJ}"

# 5️⃣ Cleanup
rm -f "${TMP_DUMP}"
