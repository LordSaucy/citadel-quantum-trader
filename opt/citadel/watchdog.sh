#!/usr/bin/env bash
# =============================================================================
# citadel/watchdog.sh
#
# Purpose
# -------
#   • Periodically verify that the primary CQT engine is healthy:
#        – HTTP health‑check (`/health`)
#        – Prometheus readiness endpoint
#        – Replication lag (PostgreSQL) via Prometheus metric
#   • If any check fails consecutively more than $MAX_FAILS times, move the
#     floating IP from the primary droplet to the standby droplet.
#   • Optional Slack webhook alerts for every state change / failure.
#
#   The script is designed to run as a **systemd service** (or a Docker
#   side‑car) on the primary droplet.  It is **idempotent** – the IP move
#   happens only once per failure episode.
#
# Configuration (environment variables – can be exported in the service file)
# -------------------------------------------------------------------------
#   HEALTH_URL          – HTTP health endpoint of the engine (default localhost)
#   PROM_URL            – Prometheus base URL (default http://localhost:9090)
#   FLOATING_IP_ID      – DigitalOcean Floating IP identifier (required)
#   PRIMARY_HOST        – Hostname or IP of the primary droplet (used for logging)
#   STANDBY_HOST        – Hostname or IP of the standby droplet
#   STANDBY_DROPLET_ID – DigitalOcean droplet ID of the standby (required)
#   MAX_FAILS           – How many consecutive failures trigger fail‑over (default 3)
#   INTERVAL            – Seconds between health checks (default 30)
#   SLACK_WEBHOOK_URL   – Optional Slack webhook URL for alerts
#   LOG_FILE            – Path to the watchdog log (default /var/log/citadel/watchdog.log)
# =============================================================================

set -euo pipefail               # abort on error, undefined vars, and pipe failures
IFS=$'\n\t'                     # sane field splitting

# ------------------------------
# 1️⃣  Configuration defaults
# ------------------------------
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
PROM_URL="${PROM_URL:-http://localhost:9090}"
FLOATING_IP_ID="${FLOATING_IP_ID:?Floating IP ID must be set}"
PRIMARY_HOST="${PRIMARY_HOST:-$(hostname)}"
STANDBY_HOST="${STANDBY_HOST:-standby}"
STANDBY_DROPLET_ID="${STANDBY_DROPLET_ID:?Standby droplet ID must be set}"
MAX_FAILS="${MAX_FAILS:-3}"
INTERVAL="${INTERVAL:-30}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
LOG_FILE="${LOG_FILE:-/var/log/citadel/watchdog.log}"

# ------------------------------
# 2️⃣  Helper: logging + optional Slack alert
# ------------------------------
log() {
    local msg=$1
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    printf '%s [watchdog] %s\n' "$ts" "$msg" | tee -a "$LOG_FILE"

    # If a Slack webhook is configured, post a short alert
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        # Escape double quotes for JSON payload
        local escaped_msg
        escaped_msg=$(printf '%s' "$msg" | sed 's/"/\\"/g')
        curl -s -X POST -H 'Content-Type: application/json' \
            -d "{\"text\":\"$ts watchdog: $escaped_msg\"}" \
            "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# ------------------------------
# 3️⃣  Helper: assign positional parameters to locals
# ------------------------------
# All functions below follow the pattern:
#   local var_name=$1   # $1 is the first positional argument
#   shift                # advance the argument list (if more args are needed)

# ------------------------------
# 4️⃣  Health‑check functions (each returns 0 on success, 1 on failure)
# ------------------------------

check_health() {
    local url=$1
    local rc
    rc=$(curl -sSf -m 5 -o /dev/null -w "%{http_code}" "$url") || rc=000
    if [[ "$rc" == "200" ]]; then
        return 0
    else
        log "Engine health check failed (HTTP $rc) – URL: $url"
        return 1
    fi
}
# Explicit return for readability (ShellCheck prefers it)
return 0

check_prometheus() {
    local url=$1
    local rc
    rc=$(curl -sSf -m 5 -o /dev/null -w "%{http_code}" "$url/-/ready") || rc=000
    if [[ "$rc" == "200" ]]; then
        return 0
    else
        log "Prometheus readiness check failed (HTTP $rc) – URL: $url"
        return 1
    fi
}
return 0

check_replication_lag() {
    local prom_base=$1
    local max_lag_seconds=${2:-5}   # default threshold = 5 s
    local metric_url lag_raw lag_sec

    metric_url="${prom_base}/api/v1/query?query=pg_replication_lag_seconds"
    lag_raw=$(curl -sSf -m 5 "$metric_url" 2>/dev/null || true)

    # Extract the value with jq; if missing, treat as failure
    lag_sec=$(printf '%s' "$lag_raw" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "")

    if [[ -z "$lag_sec" ]]; then
        log "Replication‑lag metric missing or unparsable"
        return 1
    fi

    # Compare as integers (jq can do the numeric comparison safely)
    if awk "BEGIN{exit(($lag_sec > $max_lag_seconds) ? 1 : 0)}"; then
        log "Replication lag too high: ${lag_sec}s (threshold ${max_lag_seconds}s)"
        return 1
    fi

    return 0
}
return 0

# ------------------------------
# 5️⃣  Fail‑over action – move the floating IP
# ------------------------------
move_floating_ip() {
    local floating_ip_id=$1
    local target_droplet_id=$2
    local response

    log "Attempting to move Floating IP $floating_ip_id → Droplet $target_droplet_id"

    # DigitalOcean API – you must have DO_TOKEN exported in the environment
    if [[ -z "${DO_TOKEN:-}" ]]; then
        log "ERROR: DO_TOKEN environment variable not set – cannot call DigitalOcean API"
        return 1
    fi

    response=$(curl -sSf -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $DO_TOKEN" \
        -d "{\"type\":\"assign\",\"droplet_id\":$target_droplet_id}" \
        "https://api.digitalocean.com/v2/floating_ips/${floating_ip_id}/actions") || {
        log "Failed to request Floating IP reassignment (curl error)"
        return 1
    }

    # Simple success check – DigitalOcean returns a JSON with "status":"completed" eventually.
    if printf '%s' "$response" | jq -e '.action.type == "assign"' >/dev/null; then
        log "Floating IP reassignment request accepted"
        return 0
    else
        log "Floating IP reassignment failed – response: $response"
        return 1
    fi
}
return 0

# ------------------------------
# 6️⃣  Main watchdog loop
# ------------------------------
main_loop() {
    local fail_count=0
    local ok_checks

    while true; do
        ok_checks=0

        # 1️⃣  Engine health
        if check_health "$HEALTH_URL"; then
            ((ok_checks++))
        fi

        # 2️⃣  Prometheus readiness
        if check_prometheus "$PROM_URL"; then
            ((ok_checks++))
        fi

        # 3️⃣  Replication lag
        if check_replication_lag "$PROM_URL"; then
            ((ok_checks++))
        fi

        # -------------------------------------------------
        # Decision point – did we pass *all* checks?
        # -------------------------------------------------
        if (( ok_checks == 3 )); then
            # All good – reset failure counter
            if (( fail_count > 0 )); then
                log "All services healthy again – resetting failure counter"
            fi
            fail_count=0
        else
            ((fail_count++))
            log "Health check failure #$fail_count (threshold $MAX_FAILS)"
        fi

        # -------------------------------------------------
        # Trigger fail‑over if we exceeded the allowed failures
        # -------------------------------------------------
        if (( fail_count >= MAX_FAILS )); then
            log "FAILOVER: $MAX_FAILS consecutive failures – moving floating IP to standby"

            if move_floating_ip "$FLOATING_IP_ID" "$STANDBY_DROPLET_ID"; then
                log "FAILOVER SUCCESS – floating IP now points at standby ($STANDBY_HOST)"
                # Reset counter after a successful switchover
                fail_count=0
            else
                log "FAILOVER ERROR – unable to move floating IP; will retry on next loop"
            fi
        fi

        # -------------------------------------------------
        # Sleep before the next iteration
        # -------------------------------------------------
        sleep "$INTERVAL"
    done
}
return 0

# ------------------------------
# 7️⃣  Graceful shutdown handling (SIGTERM/SIGINT)
# ------------------------------
cleanup() {
    log "Watchdog received termination signal – exiting cleanly"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ------------------------------
# 8️⃣  Entry point
# ------------------------------
log "=== Citadel Quantum Trader Watchdog started on $PRIMARY_HOST ==="
main_loop   # this function never returns (loops forever)
