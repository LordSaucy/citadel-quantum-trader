#!/usr/bin/env bash
# -------------------------------------------------
# verify_backup_restore.sh
#   • Takes the newest nightly pg_dump (custom format) from S3
#   • Restores it into a temporary PostgreSQL instance
#   • Compares row‑counts and a sample of recent trades
#   • Measures total elapsed time (RTO) and data‑loss % (RPO)
# -------------------------------------------------
set -euo pipefail

# ------------ CONFIG -------------------------------------------------
S3_BUCKET="s3://citadel-audit/backup"
TMP_DIR="/tmp/citadel_restore_$$"
RESTORE_CONTAINER="citadel-restore-test"
SOURCE_CONTAINER="citadel-db"
DB_USER="citadel"
DB_NAME="citadel"
# How many recent rows to sample for a deeper check (0 = skip)
SAMPLE_SIZE=20
# -------------------------------------------------
mkdir -p "${TMP_DIR}"
cd "${TMP_DIR}"

# 1️⃣ Grab the newest backup file from S3
echo "📥 Downloading latest backup from ${S3_BUCKET} ..."
LATEST_OBJ=$(aws s3 ls "${S3_BUCKET}/" | sort | tail -n1 | awk '{print $4}')
if [[ -z "${LATEST_OBJ}" ]]; then
  echo "❌ No backup objects found in ${S3_BUCKET}"
  exit 1
fi
aws s3 cp "${S3_BUCKET}/${LATEST_OBJ}" "./backup.dump"
echo "✅ Downloaded ${LATEST_OBJ}"

# 2️⃣ Record start time
START_TS=$(date +%s)

# 3️⃣ Restore into the temporary DB
echo "🔄 Restoring backup into container ${RESTORE_CONTAINER} ..."
docker exec -i "${RESTORE_CONTAINER}" pg_restore \
  --no-owner \
  --no-acl \
  --dbname="${DB_NAME}" \
  --username="${DB_USER}" \
  --jobs=$(nproc) \
  ./backup.dump > restore.log 2>&1

# 4️⃣ Record end time and compute RTO
END_TS=$(date +%s)
RTO_SEC=$((END_TS - START_TS))
echo "⏱️  Restore completed in ${RTO_SEC}s"

# 5️⃣ Row‑count comparison
SRC_COUNT=$(docker exec "${SOURCE_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM trades;")
TGT_COUNT=$(docker exec "${RESTORE_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM trades;")
echo "🔢 Source row count: ${SRC_COUNT}"
echo "🔢 Target row count: ${TGT_COUNT}"

# 6️⃣ Compute RPO (percentage of rows lost)
if [[ "${SRC_COUNT}" -eq 0 ]]; then
  RPO_PCT=0
else
  LOST=$((SRC_COUNT - TGT_COUNT))
  RPO_PCT=$(awk "BEGIN {printf \"%.4f\", (${LOST}/${SRC_COUNT})*100}")
fi
echo "📉 Data‑loss (RPO) = ${RPO_PCT}% (lost ${LOST} rows)"

# 7️⃣ Sample recent trades (optional but recommended)
if (( SAMPLE_SIZE > 0 )); then
  echo "🔎 Sampling the most recent ${SAMPLE_SIZE} trades from each DB ..."
  SRC_SAMPLE=$(docker exec "${SOURCE_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" \
    -t -c "SELECT id, ts, bucket_id, pnl FROM trades ORDER BY ts DESC LIMIT ${SAMPLE_SIZE};")
  TGT_SAMPLE=$(docker exec "${RESTORE_CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" \
    -t -c "SELECT id, ts, bucket_id, pnl FROM trades ORDER BY ts DESC LIMIT ${SAMPLE_SIZE};")

  # Compare line‑by‑line (order matters because we sorted by ts)
  DIFF=$(diff <(echo "${SRC_SAMPLE}") <(echo "${TGT_SAMPLE}") || true)
  if [[ -z "${DIFF}" ]]; then
    echo "✅ Sampled trades match exactly."
  else
    echo "⚠️  Sampled trades differ!"
    echo "${DIFF}"
  fi
fi

# 8️⃣ Final pass/fail logic
PASS=true
if (( RTO_SEC > 300 )); then   # 5 min = 300 s
  echo "❌ RTO exceeds 5 min (got ${RTO_SEC}s)."
  PASS=false
fi
if (( $(awk "BEGIN {print (${RPO_PCT}>1)}") )); then
  echo "❌ RPO exceeds 1 % (got ${RPO_PCT}%)."
  PASS=false
fi

if $PASS; then
  echo "🎉 BACKUP‑RESTORE VERIFICATION PASSED."
  exit 0
else
  echo "🚨 BACKUP‑RESTORE VERIFICATION FAILED."
  exit 1
fi
