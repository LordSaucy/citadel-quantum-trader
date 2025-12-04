#!/usr/bin/env bash
set -euo pipefail

BUCKET="citadel-audit"
PREFIX="db-dumps/"

# Get the latest object's LastModified timestamp
LATEST=$(aws s3api list-object-versions \
    --bucket "$BUCKET" \
    --prefix "$PREFIX" \
    --query "Versions[?IsLatest==`true`].LastModified" \
    --output text | sort -r | head -n1)

# Convert to epoch seconds
LATEST_EPOCH=$(date -d "$LATEST" +%s)
NOW_EPOCH=$(date +%s)

# If the dump is older than 36 hours, raise an alarm (you can also send to Slack)
if (( NOW_EPOCH - LATEST_EPOCH > 129600 )); then
  echo "⚠️ No recent dump in the last 36 h (latest: $LATEST)" >&2
  # Example: send to Slack via webhook
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"⚠️ Citadel backup missing – latest dump $LATEST\"}" \
    "$SLACK_WEBHOOK"
fi
