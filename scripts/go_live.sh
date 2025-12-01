#!/usr/bin/env bash
# =============================================================================
# go_live.sh – One‑click production deployment for Citadel Quantum Trader (CQT)
#
# What it does:
#   1️⃣  Loads configuration from cqt_env.sh
#   2️⃣  Pulls the latest Docker image on both engine droplets
#   3️⃣  Restarts the engine service via Docker‑Compose
#   4️⃣  Waits for the Flask health endpoint to become healthy
#   5️⃣  (Optional) Updates the DigitalOcean Load Balancer backend list
#   6️⃣  Runs the full validation harness (scripts/validate.sh)
#
# Requirements:
#   • doctl installed and authenticated (for optional LB update)
#   • SSH key that can log in as $SSH_USER on all droplets
#   • Docker & Docker‑Compose installed on the droplets
#   • Docker secrets already created on the engine droplets
#   • The repository already cloned under $DEPLOY_ROOT on each droplet
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------
# 0️⃣  Load environment
# -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/cqt_env.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ Environment file $ENV_FILE not found – aborting."
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

# -----------------------------------------------------------------
# Helper: pretty printing with timestamps
# -----------------------------------------------------------------
log() {
    local ts
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "[$ts] $*" | tee -a "$LOG_FILE"
}
err() {
    log "❌ $*"
}
ok() {
    log "✅ $*"
}
warn() {
    log "⚠️ $*"
}

# -----------------------------------------------------------------
# 1️⃣  Sanity checks – make sure required vars are set
# -----------------------------------------------------------------
required_vars=(SSH_USER SSH_KEY_PATH PRIMARY_IP STANDBY_IP FULL_IMAGE DEPLOY_ROOT LOG_FILE)
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        err "Required variable $var is empty – check cqt_env.sh"
        exit 1
    fi
done

# Verify SSH key exists
if [[ ! -f "$SSH_KEY_PATH" ]]; then
    err "SSH key $SSH_KEY_PATH not found"
    exit 1
fi

# -----------------------------------------------------------------
# 2️⃣  Function to run a remote command via SSH
# -----------------------------------------------------------------
ssh_cmd() {
    local host="$1"
    shift
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "${SSH_USER}@${host}" "$@"
}

# -----------------------------------------------------------------
# 3️⃣  Pull the latest image on a given droplet
# -----------------------------------------------------------------
pull_image() {
    local host="$1"
    log "Pulling image $FULL_IMAGE on $host ..."
    ssh_cmd "$host" "cd $DEPLOY_ROOT && docker compose pull engine"
    ok "Image pulled on $host"
}

# -----------------------------------------------------------------
# 4️⃣  Restart the engine service on a given droplet
# -----------------------------------------------------------------
restart_engine() {
    local host="$1"
    log "Restarting engine service on $host ..."
    ssh_cmd "$host" "cd $DEPLOY_ROOT && docker compose up -d --force-recreate engine"
    ok "Engine restarted on $host"
}

# -----------------------------------------------------------------
# 5️⃣  Wait for the Flask health endpoint to become healthy
# -----------------------------------------------------------------
wait_for_health() {
    local host="$1"
    local timeout=60   # seconds
    local elapsed=0
    local url="http://$host:8005/healthz"

    log "Waiting for health endpoint $url (max $timeout s)..."
    while (( elapsed < timeout )); do
        if ssh_cmd "$host" "curl -sSf $url" >/dev/null 2>&1; then
            ok "Health endpoint on $host is UP"
            return 0
        fi
        sleep 3
        (( elapsed += 3 ))
    done
    err "Health endpoint on $host never became ready (timeout $timeout s)"
    return 1
}

# -----------------------------------------------------------------
# 6️⃣  (Optional) Update DigitalOcean Load Balancer backend list
# -----------------------------------------------------------------
update_lb_backends() {
    if [[ -z "${LB_ID:-}" ]]; then
        warn "LB_ID not set – skipping Load Balancer backend update"
        return 0
    fi

    log "Updating Load Balancer ($LB_ID) backend list..."

    # Gather the droplet IDs for the two engine hosts
    ENGINE_IDS=$(doctl compute droplet list --tag-name cqt,engine \
                 --format ID --no-header | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$ENGINE_IDS" ]]; then
        err "Could not find engine droplet IDs – aborting LB update"
        return 1
    fi

    doctl compute load-balancer update "$LB_ID" --droplet-ids "$ENGINE_IDS"
    ok "Load Balancer backends refreshed"
}

# -----------------------------------------------------------------
# 7️⃣  Run the validation harness (scripts/validate.sh)
# -----------------------------------------------------------------
run_validation() {
    local validator="${SCRIPT_DIR}/validate.sh"
    if [[ ! -x "$validator" ]]; then
        err "Validation script $validator not found or not executable"
        return 1
    fi
    log "Running post‑deployment validation..."
    "$validator"
    ok "Validation completed successfully"
}

# -----------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------
log "===== STARTING CQT GO‑LIVE DEPLOYMENT ====="
log "Log file: $LOG_FILE"

# Pull image on both droplets (parallel to save time)
log "=== Pulling latest image on both engines ==="
pull_image "$PRIMARY_IP" &
PID_PRIMARY=$!
pull_image "$STANDBY_IP" &
PID_STANDBY=$!
wait $PID_PRIMARY $PID_STANDBY
ok "Image pulled on both engine droplets"

# Restart engine services
log "=== Restarting engine containers ==="
restart_engine "$PRIMARY_IP" &
PID_PRIMARY=$!
restart_engine "$STANDBY_IP" &
PID_STANDBY=$!
wait $PID_PRIMARY $PID_STANDBY
ok "Engine containers restarted on both droplets"

# Wait for health endpoints (serial – we need both up before proceeding)
log "=== Waiting for health checks ==="
wait_for_health "$PRIMARY_IP"
wait_for_health "$STANDBY_IP"

# Optional: refresh the Load Balancer backend list
update_lb_backends

# Run the full validation suite
run_validation

log "===== CQT DEPLOYMENT FINISHED SUCCESSFULLY ====="
exit 0
