#!/usr/bin/env bash
# =============================================================================
#  rollback_to_previous.sh
#
#  Production‑ready rollback script for the Citadel Quantum Trader (CQT)
#  engine.  It pulls a previous Docker image (by tag) from the container
#  registry and restarts the engine containers on both primary and standby
#  droplets.
#
#  Usage:
#      ./rollback_to_previous.sh               # auto‑detect previous tag
#      ./rollback_to_previous.sh v1.2.3        # roll back to explicit tag
#
#  Requirements:
#      * docker‑compose.yml must reference ${IMAGE_TAG}
#      * SSH key‑based access to both droplets
#      * GITHUB_TOKEN env‑var with repo scope (for tag lookup)
# =============================================================================

set -euo pipefail

# -------------------------------------------------------------------------
# Helper functions
# ✅ FIXED: Added explicit return statements to all functions
# -------------------------------------------------------------------------
log() {
    echo -e "\e[32m[+] $*\e[0m"
    return 0
}

warn() {
    echo -e "\e[33m[!] $*\e[0m"
    return 0
}

error() {
    echo -e "\e[31m[✖] $*\e[0m" >&2
    return 1
}

die() {
    error "$*"
    exit 1
    # Note: exit 1 terminates the script, so this return is unreachable,
    # but SonarCloud S1871 requires explicit returns in all code paths
    return 1
}

# -------------------------------------------------------------------------
# Load environment (you can source a .env file before invoking)
# -------------------------------------------------------------------------
: "${PRIMARY_IP:?Missing PRIMARY_IP}"
: "${STANDBY_IP:?Missing STANDBY_IP}"
: "${SSH_USER:=root}"
: "${SSH_KEY:=$HOME/.ssh/id_rsa}"
: "${GITHUB_REPOSITORY_OWNER:?Missing GITHUB_REPOSITORY_OWNER}"
: "${IMAGE_REPO:=ghcr.io/${GITHUB_REPOSITORY_OWNER}/cqt-engine}"
: "${GITHUB_TOKEN:?Missing GITHUB_TOKEN}"

# -------------------------------------------------------------------------
# Determine which tag to roll back to
# -------------------------------------------------------------------------
TARGET_TAG="${1:-}"   # optional first argument

if [[ -z "$TARGET_TAG" ]]; then
    log "No tag supplied – fetching the previous Git tag from the repo..."

    # Get all tags, sort by version‑aware order, keep the two newest
    # (the newest is the current production tag, the second is the previous)
    TAGS=$(git ls-remote --tags "https://github.com/${GITHUB_REPOSITORY_OWNER}/citadel-quantum-trader.git" |
          awk -F'[/]' '{print $NF}' |
          grep -v '^{}$' |
          sort -V |
          tail -n2)

    # The list now looks like:
    #   prod-2024.11.26
    #   prod-2024.11.19   <-- we want this one
    PREV_TAG=$(echo "$TAGS" | head -n1)

    if [[ -z "$PREV_TAG" ]]; then
        die "Unable to determine a previous tag – aborting."
    fi

    TARGET_TAG="$PREV_TAG"
    log "Selected previous tag: $TARGET_TAG"
else
    log "Explicit tag supplied: $TARGET_TAG"
fi

# -------------------------------------------------------------------------
# Function to run a command on a remote droplet via SSH
# ✅ FIXED: Added explicit return statement
# -------------------------------------------------------------------------
remote_exec() {
    local host=$1
    shift
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${SSH_USER}@${host}" "$@"
    return $?
}

# -------------------------------------------------------------------------
# Pull the image and restart the engine on a single droplet
# ✅ FIXED: Added explicit return statements to all code paths
# -------------------------------------------------------------------------
rollback_one() {
    local host=$1
    log "Rolling back engine on $host to tag $TARGET_TAG …"

    # 1️⃣ Pull the exact image tag
    if ! remote_exec "$host" "docker pull ${IMAGE_REPO}:${TARGET_TAG}"; then
        error "Failed to pull image on $host"
        return 1
    fi

    # 2️⃣ Export IMAGE_TAG env var for docker‑compose (temporarily)
    #    We use a tiny wrapper script on the remote host so we don't
    #    have to edit the compose file permanently.
    if ! remote_exec "$host" bash -c "
        export IMAGE_TAG=${TARGET_TAG}
        cd /opt/cqt
        docker compose up -d --no-deps --force-recreate engine
    "; then
        error "Failed to restart engine on $host"
        return 1
    fi

    # 3️⃣ Verify the container is healthy
    log "Waiting for health endpoint on $host …"
    for i in {1..12}; do   # up to 60 seconds
        STATUS=$(remote_exec "$host" curl -s -o /dev/null -w "%{http_code}" http://localhost:8005/healthz || echo "000")
        if [[ "$STATUS" == "200" ]]; then
            log "Engine on $host reports healthy (HTTP 200)."
            return 0
        fi
        sleep 5
    done

    error "Engine on $host never became healthy after rollback."
    return 1
}

# -------------------------------------------------------------------------
# Perform rollback on both droplets
# -------------------------------------------------------------------------
log "=== Starting rollback to tag ${TARGET_TAG} ==="

if ! rollback_one "$PRIMARY_IP"; then
    die "Rollback on PRIMARY_IP ($PRIMARY_IP) failed."
fi

if ! rollback_one "$STANDBY_IP"; then
    die "Rollback on STANDBY_IP ($STANDBY_IP) failed."
fi

# -------------------------------------------------------------------------
# Run the full validation harness (scripts/validate.sh)
# -------------------------------------------------------------------------
log "Running post‑rollback validation..."
if ./scripts/validate.sh; then
    log "✅ Validation passed – rollback successful!"
    exit 0
else
    die "❌ Validation failed after rollback.  Investigate the logs!"
fi
