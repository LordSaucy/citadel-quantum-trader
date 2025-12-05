#!/usr/bin/env bash
# =============================================================================
# Citadel Quantum Trader – Full‑HA Watchdog & Automatic Fail‑Over
# =============================================================================
# What it does (in order):
#   1️⃣  Periodically checks three health indicators:
#        • FastAPI health endpoint
#        • Prometheus readiness endpoint
#        • PostgreSQL replication lag (via Prometheus)
#   2️⃣  If ANY indicator fails for MAX_FAILURES consecutive checks → trigger fail‑over.
#   3️⃣  Move the floating IP to the *other* droplet.
#   4️⃣  Promote the PostgreSQL replica on the new primary (if needed).
#   5️⃣  Pull latest Docker images and restart the full Docker‑Compose stack.
#   6️⃣  Update droplet tags so the next fail‑over knows which is primary/standby.
#   7️⃣  Reset the failure counter and continue monitoring.
#
# The script is written for **DigitalOcean** (`doctl`).  Replace the
# `move_floating_ip()` function with the equivalent CLI/API calls for AWS,
# Hetzner, Linode, etc.
# =============================================================================

# --------------------------- CONFIGURATION -------------------------------
FLOAT_IP="203.0.113.42"                # Your floating IP address
PRIMARY_TAG="citadel-primary"          # Tag identifying the primary droplet
STANDBY_TAG="citadel-standby"          # Tag identifying the standby droplet

HEALTH_URL="http://127.0.0.1:8000/health"   # CQT FastAPI health endpoint (local)
PROM_URL="http://127.0.0.1:9090"            # Prometheus endpoint (local)

MAX_FAILURES=3                         # Consecutive failures before fail‑over
INTERVAL=30                            # Seconds between health checks

SSH_KEY="${HOME}/.ssh/id_rsa"          # SSH private key for root login
REMOTE_USER="root"                     # User for SSH (Vultr droplets default to root)

# Optional alerting (Slack, Teams, Discord, etc.)
WEBHOOK="${SLACK_WEBHOOK_URL:-}"       # Set env var SLACK_WEBHOOK_URL if you want alerts

# --------------------------- HELPERS ------------------------------------
# All helper functions now end with an explicit `return` so SonarQube
# recognises a deterministic exit status.

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    # Write to local logfile and also echo to stdout (tee)
    echo "$ts [watchdog] $*" | tee -a /var/log/citadel/watchdog.log

    # Optional webhook notification (e.g., Slack)
    if [[ -n "$WEBHOOK" ]]; then
        curl -s -X POST -H 'Content-Type: application/json' \
            -d "{\"text\":\"$ts watchdog: $*\"}" "$WEBHOOK" >/dev/null 2>&1
    fi
    return 0
}

# ---- Provider‑specific floating‑IP move (DigitalOcean) -----------------
move_floating_ip() {
    local target_id=$1
    log "Moving floating IP $FLOAT_IP → droplet $target_id"

    # ---- DIGITALOCEAN -------------------------------------------------
    if ! doctl compute floating-ip-action assign "$FLOAT_IP" "$target_id"; then
        log "❌  Failed to assign floating IP via doctl."
        return 1
    fi

    # ---- If you use another provider, replace the line above with the
    #      equivalent CLI/API call (AWS Elastic IP, Hetzner Floating IP, etc.)
    return 0
}

# ---- Simple HTTP health check – returns HTTP status code ----------
check_http() {
    local url=$1
    # -s silent, -o discard body, -w output only the HTTP code
    curl -s -o /dev/null -w "%{http_code}" "$url"
    return 0
}

# ---- FastAPI health -------------------------------------------------
check_fastapi() {
    local rc
    rc=$(check_http "$HEALTH_URL")
    if [[ "$rc" == "200" ]]; then
        return 0
    fi
    return 1
}

# ---- Prometheus readiness -------------------------------------------
check_prometheus() {
    local rc
    rc=$(check_http "$PROM_URL/-/ready")
    if [[ "$rc" == "200" ]]; then
        return 0
    fi
    return 1
}

# ---- Replication lag (via Prometheus) -------------------------------
# Returns true (0) if lag < 5 seconds, false (1) otherwise.
check_replication_lag() {
    local lag
    lag=$(curl -s "$PROM_URL/api/v1/query?query=pg_replication_lag_seconds" \
        | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "")
    if [[ -z "$lag" ]]; then
        log "⚠️  Replication lag metric unavailable."
        return 1
    fi
    # Fail if lag > 5 seconds
    awk "BEGIN{exit ($lag > 5)}"
    return $?
}

# -----------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------
fail_count=0

while true; do
    ok=0

    # Each check returns 0 on success, 1 on failure.
    if check_fastapi;    then ((ok++)); fi
    if check_prometheus; then ((ok++)); fi
    if check_replication_lag; then ((ok++)); fi

    if (( ok < 3 )); then
        ((fail_count++))
        log "Health check FAILED (ok=$ok). Failure #$fail_count"
    else
        if (( fail_count > 0 )); then
            log "Health check PASSED – resetting failure counter."
        fi
        fail_count=0
    fi

    # ---------------------------------------------------------------
    # Trigger fail‑over once we exceed the allowed number of failures
    # ---------------------------------------------------------------
    if (( fail_count >= MAX_FAILURES )); then
        log "=== FAIL‑OVER INITIATED ==="

        # ----------- 1️⃣ Identify current floating‑IP owner -----------
        CURRENT_OWNER=$(doctl compute floating-ip get "$FLOAT_IP" \
                         --format DropletID --no-header)

        # ----------- 2️⃣ Resolve droplet IDs for primary & standby ----
        PRIMARY_ID=$(doctl compute droplet list --tag-name "$PRIMARY_TAG" \
                         --format ID --no-header | head -n1)
        STANDBY_ID=$(doctl compute droplet list --tag-name "$STANDBY_TAG" \
                         --format ID --no-header | head -n1)

        if [[ -z "$PRIMARY_ID" || -z "$STANDBY_ID" ]]; then
            log "ERROR: Could not resolve droplet IDs for tags $PRIMARY_TAG / $STANDBY_TAG"
            exit 1
        fi

        # ----------- 3️⃣ Decide which node becomes the new primary ----
        if [[ "$CURRENT_OWNER" == "$PRIMARY_ID" ]]; then
            # Primary currently holds the floating IP → move it to standby
            TARGET_ID=$STANDBY_ID
            NEW_PRIMARY_TAG=$STANDBY_TAG
            NEW_STANDBY_TAG=$PRIMARY_TAG
        else
            # Standby already holds the floating IP → move it to primary
            TARGET_ID=$PRIMARY_ID
            NEW_PRIMARY_TAG=$PRIMARY_TAG
            NEW_STANDBY_TAG=$STANDBY_TAG
        fi

        # ----------- 4️⃣ Move the floating IP ------------------------
        if ! move_floating_ip "$TARGET_ID"; then
            log "❌  Fail‑over aborted – could not move floating IP."
            # Do NOT reset fail_count; we’ll retry on the next loop.
            sleep "$INTERVAL"
            continue
        fi

        # ----------- 5️⃣ Promote PostgreSQL on the new primary -------
        NEW_PRIMARY_IP=$(doctl compute droplet get "$TARGET_ID" \
                         --format PrivateIPv4 --no-header)
        log "Promoting PostgreSQL replica on new primary ($NEW_PRIMARY_IP)…"
        ssh -i "$SSH_KEY" "$REMOTE_USER@$NEW_PRIMARY_IP" bash -s <<'EOSSH'
docker exec -i citadel-db-standby pg_ctl -D /var/lib/postgresql/data -w promote
EOSSH
        # If the promotion fails, the remote command will exit non‑zero and the script will abort.
        if [[ $? -ne 0 ]]; then
            log "❌  PostgreSQL promotion failed."
            sleep "$INTERVAL"
            continue
        fi

        # ----------- 6️⃣ Restart the Docker‑Compose stack -------------
        log "Restarting CQT Docker stack on $NEW_PRIMARY_IP…"
        ssh -i "$SSH_KEY" "$REMOTE_USER@$NEW_PRIMARY_IP" bash -s <<'EOSSH'
cd /opt/cqt
docker compose pull          # optional but recommended
docker compose up -d        # start / restart all services
EOSSH

        # ----------- 7️⃣ Update droplet tags -------------------------
        log "Updating droplet tags: $NEW_PRIMARY_TAG ← primary, $NEW_STANDBY_TAG ← standby"
        # Remove old tags
        doctl compute droplet remove-tag "$PRIMARY_ID" "$PRIMARY_TAG"
        doctl compute droplet remove-tag "$STANDBY_ID" "$STANDBY_TAG"
        # Apply new tags
        doctl compute droplet add-tag "$TARGET_ID" "$NEW_PRIMARY_TAG"
        OTHER_ID=$(( TARGET_ID == PRIMARY_ID ? STANDBY_ID : PRIMARY_ID ))
        doctl compute droplet add-tag "$OTHER_ID" "$NEW_STANDBY_TAG"

        # ----------- 8️⃣ Reset failure counter -----------------------
        fail_count=0
        log "=== FAIL‑OVER COMPLETE === New primary: $NEW_PRIMARY_TAG (droplet $TARGET_ID) ==="
    fi

    sleep "$INTERVAL"
done
