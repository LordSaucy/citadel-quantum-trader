#!/usr/bin/env bash
# -------------------------------------------------
# restore_test.sh
# 1️⃣ Pull the latest dump & its checksum
# 2️⃣ Spin up a temporary PostgreSQL container
# 3️⃣ Restore the dump
# 4️⃣ Run validation (row‑counts + schema compare)
# 5️⃣ Write a JSON status file to S3 (+ optional Slack alert)
# -------------------------------------------------
set -euo pipefail

# ==== CONFIG ==============================================================
BUCKET="citadel-audit"
DUMP_PREFIX="backup/"
HASH_SUFFIX="${DUMP_PREFIX}hashes/"
VERIF_PREFIX="backup_verification/"   # where we store JSON status files
TEMP_CONTAINER_NAME="citadel-restore-test"
PG_PASSWORD="temp_pass_$(openssl rand -hex 8)"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"   # set in env if you want Slack alerts
# ========================================================================

# Helper: send a Slack message (if webhook is defined)
slack_notify() {
  local msg="$1"
  if [[ -n "${SLACK_WEBHOOK}" ]]; then
    curl -s -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"${msg}\"}" "${SLACK_WEBHOOK}" >/dev/null
  fi
}

# 1️⃣ Identify latest dump & checksum
LATEST_DUMP=$(aws s3 ls "s3://${BUCKET}/${DUMP_PREFIX}" --recursive \
               | sort -k1,2 | tail -n1 | awk '{print $4}')
HASH_OBJ="${HASH_SUFFIX}$(basename "${LATEST_DUMP}").sha256"

if [[ -z "${LATEST_DUMP}" ]]; then
  echo "❌ No dump found"
  exit 1
fi

echo "🔎 Dump to test: ${LATEST_DUMP}"
echo "🔎 Expected checksum object: ${HASH_OBJ}"

# 2️⃣ Download dump & checksum
TMP_DUMP="/tmp/$(basename "${LATEST_DUMP}")"
TMP_HASH="/tmp/$(basename "${HASH_OBJ}")"
aws s3 cp "s3://${BUCKET}/${LATEST_DUMP}" "${TMP_DUMP}"
aws s3 cp "s3://${BUCKET}/${HASH_OBJ}" "${TMP_HASH}"

# Extract the stored checksum (first field)
EXPECTED_SUM=$(awk '{print $1}' "${TMP_HASH}")

# Compute local checksum
LOCAL_SUM=$(sha256sum "${TMP_DUMP}" | awk '{print $1}')

if [[ "${EXPECTED_SUM}" != "${LOCAL_SUM}" ]]; then
  MSG="❌ Checksum mismatch for ${LATEST_DUMP}! Expected ${EXPECTED_SUM}, got ${LOCAL_SUM}"
  echo "${MSG}"
  slack_notify "${MSG}"
  exit 1
fi
echo "✅ Checksum verified"

# 3️⃣ Spin up a temporary PostgreSQL container (clean slate)
docker rm -f "${TEMP_CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${TEMP_CONTAINER_NAME}" \
  -e POSTGRES_PASSWORD="${PG_PASSWORD}" \
  -e POSTGRES_DB=restore_test \
  -p 5433:5432 postgres:15-alpine >/dev/null

# Wait for DB to be ready (max 30 s)
for i in {1..30}; do
  if docker exec "${TEMP_CONTAINER_NAME}" pg_isready -U postgres >/dev/null 2>&1; then
    echo "✅ Temporary DB ready"
    break
  fi
  sleep 1
done

# 4️⃣ Restore the dump
docker exec -i "${TEMP_CONTAINER_NAME}" pg_restore \
  -U postgres -d restore_test -Fc < "${TMP_DUMP}"

echo "✅ Dump restored into temporary DB"

# 5️⃣ Validation ---------------------------------------------------------

# 5a – Row‑count per table (source vs restored)
# First, get a list of tables from the source DB (the *live* DB)
LIVE_CONN="postgresql://citadel:${PG_PASSWORD}@citadel-db:5432/citadel"
TABLES=$(psql "${LIVE_CONN}" -Atc "SELECT tablename FROM pg_tables WHERE schemaname='public';")

VALIDATION_PASS=true
RESULT_JSON="{\"date\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",\"dump\":\"${LATEST_DUMP}\",\"checks\":[]}"

for tbl in ${TABLES}; do
  LIVE_COUNT=$(psql "${LIVE_CONN}" -Atc "SELECT COUNT(*) FROM \"${tbl}\";")
  RESTORE_COUNT=$(docker exec "${TEMP_CONTAINER_NAME}" psql -U postgres -d restore_test -Atc "SELECT COUNT(*) FROM \"${tbl}\";")
  CHECK_PASSED=true
  if [[ "${LIVE_COUNT}" -ne "${RESTORE_COUNT}" ]]; then
    CHECK_PASSED=false
    VALIDATION_PASS=false
  fi
  RESULT_JSON=$(echo "${RESULT_JSON}" | jq ".checks += [{\"table\":\"${tbl}\",\"live\":${LIVE_COUNT},\"restore\":${RESTORE_COUNT},\"pass\":${CHECK_PASSED}}]")
done

# 5b – Schema‑only dump compare (optional but cheap)
SCHEMA_DUMP="/tmp/schema_source.sql"
SCHEMA_RESTORE="/tmp/schema_restore.sql"

pg_dump -s -U citadel -d citadel -f "${SCHEMA_DUMP}"
docker exec "${TEMP_CONTAINER_NAME}" pg_dump -s -U postgres -d restore_test -f "/tmp/schema_restore.sql"
docker cp "${TEMP_CONTAINER_NAME}:/tmp/schema_restore.sql" "${SCHEMA_RESTORE}"

SCHEMA_DIFF=$(diff -u "${SCHEMA_DUMP}" "${SCHEMA_RESTORE}" || true)
if [[ -n "${SCHEMA_DIFF}" ]]; then
  VALIDATION_PASS=false
  RESULT_JSON=$(echo "${RESULT_JSON}" | jq ".checks += [{\"schema_match\":false}]")
else
  RESULT_JSON=$(echo "${RESULT_JSON}" | jq ".checks += [{\"schema_match\":true}]")
fi

# 6️⃣ Write status JSON to S3
STATUS_OBJ="${VERIF_PREFIX}$(date -u +"%Y%m%dT%H%M%SZ").json"
echo "${RESULT_JSON}" > /tmp/status.json
aws s3 cp /tmp/status.json "s3://${BUCKET}/${STATUS_OBJ}"

# 7️⃣ Final reporting
if [[ "${VALIDATION_PASS}" = true ]]; then
  MSG="✅ Backup verification succeeded for ${LATEST_DUMP}"
  echo "${MSG}"
  slack_notify "${MSG}"
else
  MSG="❌ Backup verification FAILED for ${LATEST_DUMP}. See ${STATUS_OBJ} for details."
  echo "${MSG}"
  slack_notify "${MSG}"
fi

# 8️⃣ Cleanup
docker rm -f "${TEMP_CONTAINER_NAME}" >/dev/null 2>&1
rm -f "${TMP_DUMP}" "${TMP_HASH}" "${SCHEMA_DUMP}" "${SCHEMA_RESTORE}" /tmp/status.json
