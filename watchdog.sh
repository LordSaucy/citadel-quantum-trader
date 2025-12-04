#!/usr/bin/env bash
# -------------------------------------------------
# Citadel Quantum Trader – Watchdog (fail‑over)
# -------------------------------------------------
# Configurable vars – edit to match your environment
HEALTH_URL="http://localhost:8000/health"
PROM_URL="http://prometheus:9090"
FLOATING_IP_ID="YOUR_FLOATING_IP_ID"
PRIMARY_HOST="primary-vps-hostname-or-ip"
STANDBY_HOST="standby-vps-hostname-or-ip"
STANDBY_DROPLET_ID="YOUR_STANDBY_CLOUD_ID"
MAX_FAILS=3
INTERVAL=30   # seconds between checks
WEBHOOK="${SLACK_WEBHOOK_URL:-}"   # optional Slack webhook for alerts

fail_count=0

log() {
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$ts [watchdog] $1" | tee -a /var/log/citadel/watchdog.log
  [[ -n "$WEBHOOK" ]] && curl -s -X POST -H 'Content-Type: application/json' \
    -d "{\"text\":\"$ts watchdog: $1\"}" "$WEBHOOK" >/dev/null
}

check_health() {
  rc=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")
  [[ "$rc" == "200" ]]
}

check_prometheus() {
  # Simple query to ensure Prometheus is up
  rc=$(curl -s -o /dev/null -w "%{http_code}" "$PROM_URL/-/ready")
  [[ "$rc" == "200" ]]
}

check_replication_lag() {
  # Returns true if lag < 5 seconds
  lag=$(curl -s "$PROM_URL/api/v1/query?query=pg_replication_lag_seconds" \
    | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "")
  [[ -z "$lag" ]] && return 1   # no metric → fail
  awk "BEGIN{exit ($lag > 5)}"   # fail if >5 s
}

# -------------------------------------------------
# Main loop
# -------------------------------------------------
while true; do
  ok=0
  check_health && ok=$((ok+1))
  check_prometheus && ok=$((ok+1))
  check_replication_lag && ok=$((ok+1))

  if (( ok < 3 )); then
    ((


