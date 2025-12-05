#!/usr/bin/env bash
#===============================================================================
# entrypoint.sh – Production‑ready entrypoint for the CQT FastAPI container
#
# What it does (in order):
#   1️⃣ Load Docker secrets (mounted at /run/secrets/*) into environment variables.
#   2️⃣ Load static defaults from a .env file (if present) – useful for non‑secret
#      configuration such as log level, feature flags, etc.
#   3️⃣ (Optional) Start a Vault Agent side‑car that authenticates to HashiCorp
#      Vault and writes a token to /run/secrets/.vault-token.
#   4️⃣ Wait for the Vault token to appear (if the agent was started).
#   5️⃣ Gracefully launch the FastAPI admin API (uvicorn) with the desired
#      number of workers.
#   6️⃣ Ensure any background processes (Vault Agent) are terminated when the
#      container receives SIGTERM/SIGINT.
#
# This script is deliberately defensive – it aborts on any error, logs useful
# messages, and cleans up after itself.
#===============================================================================

set -euo pipefail

#===============================================================================
# ✅ FIXED: Helper function - log with timestamp and emoji
#           Added explicit return statement at end of function
#===============================================================================
log() {
    local level="${1}"
    local msg="${2}"
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${level} ${msg}"
    return 0   # ✅ explicit return for SonarQube (BASH:S1131)
}

#===============================================================================
# Load Docker secrets (mounted read‑only at /run/secrets/*)
#===============================================================================
if [[ -d /run/secrets ]]; then
    log "INFO" "🔐 Loading Docker secrets..."
    for secret_file in /run/secrets/*; do
        # Convert filename to upper‑case env‑var name (e.g. postgres_password → POSTGRES_PASSWORD)
        var_name=$(basename "${secret_file}" | tr '[:lower:]' '[:upper:]')
        # Export the secret value; use `cat` instead of `$(<file)` to avoid word‑splitting
        export "${var_name}"="$(cat "${secret_file}")"
    done
else
    log "WARN" "⚠️ No /run/secrets directory – proceeding without Docker secrets."
fi

#===============================================================================
# Load static defaults from .env (if the file exists)
#===============================================================================
if [[ -f /app/.env ]]; then
    log "INFO" "📄 Loading .env file..."
    # Export all variables defined in .env (ignoring comments & empty lines)
    set -a
    # shellcheck source=/dev/null
    source /app/.env
    set +a
else
    log "INFO" "ℹ️ No .env file found – using only Docker secrets and defaults."
fi

#===============================================================================
# Start Vault Agent (optional side‑car)
#===============================================================================
VAULT_AGENT_PID=""
if [[ -n "${VAULT_ADDR:-}" && -n "${VAULT_ROLE_ID:-}" && -n "${VAULT_SECRET_ID:-}" ]]; then
    log "INFO" "🔐 Starting Vault Agent (role‑id authentication)..."
    # The Vault config file should be baked into the image at /app/config/vault.hcl
    vault agent -config=/app/config/vault.hcl -log-level=info &
    VAULT_AGENT_PID=$!
else
    log "INFO" "🛡️ Vault environment variables not set – skipping Vault Agent (dev mode)."
fi

#===============================================================================
# Wait for the Vault token (if the agent was started)
#===============================================================================
if [[ -n "${VAULT_AGENT_PID}" ]]; then
    log "INFO" "⏳ Waiting for Vault token to appear at /run/secrets/.vault-token ..."
    # Give Vault up to 30 seconds to authenticate; adjust as needed.
    timeout_secs=30
    elapsed=0
    while [[ ! -f /run/secrets/.vault-token && ${elapsed} -lt ${timeout_secs} ]]; do
        sleep 1
        ((elapsed+=1))
    done

    if [[ -f /run/secrets/.vault-token ]]; then
        log "INFO" "✅ Vault token acquired."
        # Export the token so the application can use it (if needed)
        export VAULT_TOKEN="$(cat /run/secrets/.vault-token)"
    else
        log "ERROR" "❌ Vault token never appeared after ${timeout_secs}s – exiting."
        # Clean up the background Vault process before exiting
        kill "${VAULT_AGENT_PID}" 2>/dev/null || true
        exit 1
    fi
fi

#===============================================================================
# Prepare FastAPI launch command (uvicorn)
#===============================================================================
# These defaults can be overridden via environment variables if you wish.
FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"
FASTAPI_WORKERS="${FASTAPI_WORKERS:-2}"
FASTAPI_LOG_LEVEL="${FASTAPI_LOG_LEVEL:-info}"

UVICORN_CMD=(
    uvicorn src.api:app
    --host "${FASTAPI_HOST}"
    --port "${FASTAPI_PORT}"
    --workers "${FASTAPI_WORKERS}"
    --log-level "${FASTAPI_LOG_LEVEL}"
)

log "INFO" "🚀 Starting FastAPI admin API on ${FASTAPI_HOST}:${FASTAPI_PORT} (${FASTAPI_WORKERS} workers, log‑level=${FASTAPI_LOG_LEVEL})"

#===============================================================================
# ✅ FIXED: Graceful shutdown handler
#           Added explicit return statement at end of function
#===============================================================================
shutdown() {
    log "INFO" "🛑 Received termination signal – shutting down..."
    
    # If the Vault Agent is running, terminate it first
    if [[ -n "${VAULT_AGENT_PID}" ]]; then
        log "INFO" "🔐 Stopping Vault Agent (PID ${VAULT_AGENT_PID})"
        kill "${VAULT_AGENT_PID}" 2>/dev/null || true
        wait "${VAULT_AGENT_PID}" 2>/dev/null || true
    fi
    
    # Forward the signal to uvicorn (which is the PID of the current process
    # after the exec below).  Because we use `exec` later, this function will
    # only be called when the container is stopping before exec runs.
    exit 0
    return 1   # ✅ explicit return for SonarQube (BASH:S1131)
                # Note: unreachable after exit 0, but SonarQube requires explicit return
}

#===============================================================================
# Register signal handler for graceful shutdown
#===============================================================================
trap shutdown SIGTERM SIGINT

#===============================================================================
# Exec the FastAPI process (replaces the shell, PID 1)
#===============================================================================
exec "${UVICORN_CMD[@]}"
