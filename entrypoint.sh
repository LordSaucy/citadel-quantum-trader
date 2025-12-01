#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Load Docker secrets (mounted at /run/secrets/*) into env vars
# ------------------------------------------------------------
if [[ -d /run/secrets ]]; then
  for secret_file in /run/secrets/*; do
    var_name=$(basename "$secret_file" | tr '[:lower:]' '[:upper:]')
    export "$var_name"="$(cat "$secret_file")"
  done
fi

# ------------------------------------------------------------
# Load static defaults from .env (if present)
# ------------------------------------------------------------
if [[ -f /app/.env ]]; then
  set -a
  source /app/.env
  set +a
fi

# ------------------------------------------------------------
# Finally exec the CMD from the Dockerfile
# ------------------------------------------------------------
exec "$@"
