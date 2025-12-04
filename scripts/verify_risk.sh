#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking DB tables ==="
docker exec -i citadel-db psql -U citadel -d citadel -c "\dt"

echo "=== Global risk metric ==="
curl -s http://localhost:9090/api/v1/query?query=global_risk_percentage | jq .

echo "=== Edge‑decay counter ==="
curl -s http://localhost:9090/api/v1/query?query=edge_decay_events_total | jq .

echo "=== Bucket risk fractions ==="
docker exec -i citadel-db psql -U citadel -d citadel -c "SELECT * FROM bucket_meta ORDER BY bucket_id;"
docker exec -i citadel-db psql -U citadel -d citadel -c "SELECT * FROM risk_modifier ORDER BY bucket_id;"

echo "=== All checks passed ==="
