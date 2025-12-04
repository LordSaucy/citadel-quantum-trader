# ------------------------------------------------------------
# 1️⃣  Builder stage – compile / install everything we need
# ------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY entrypoint.sh /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# ---- OS‑level build dependencies (gcc, libpq‑dev, etc.) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Create a non‑root user for the build (helps with file ownership) ----
ARG BUILD_UID=1000
ARG BUILD_USER=builder
RUN useradd -u ${BUILD_UID} -m ${BUILD_USER}

# ---- Working directory for the builder ----
WORKDIR /src

# ---- Copy only the files needed for dependency resolution ----
# (this allows Docker to cache the layer if requirements.txt hasn't changed)
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.cfg .
COPY src/ src/          # copy source tree – needed for editable install (optional)

# ---- Install Python dependencies into a virtual‑env‑like directory ----
# We install into /opt/venv so we can copy the whole env later.
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel && \
    ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 2️⃣  Runtime stage – tiny image that only contains the runtime env
# ------------------------------------------------------------
FROM python:3.12-slim AS runtime

# ---- Create a non‑root user that will run the container ----
ARG APP_UID=1000
ARG APP_USER=cqt
RUN useradd -u ${APP_UID} -m ${APP_USER}

# ---- Copy the virtual environment from the builder ----
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---- Copy the application source (only what we need at runtime) ----
WORKDIR /app
COPY src/ src/
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    chown -R ${APP_USER}:${APP_USER} /app

# ---- Expose the two ports used by the engine ----
# 8005 – Flask ConfluenceController API (TLS terminated at the LB)
# 8000 – Prometheus metrics endpoint (scraped by Prometheus)
EXPOSE 8005 8000

# ---- Switch to the non‑root user ----
USER ${APP_USER}

# ------------------------------------------------------------
# 3️⃣  Entrypoint – inject Docker secrets, then start the app
# ------------------------------------------------------------
# The entrypoint reads any files mounted at /run/secrets/* (Docker secrets)
# and exports them as environment variables.  It then execs the command
# passed via CMD (Gunicorn in this case).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# ------------------------------------------------------------
# 4️⃣  Default command – run the Flask app with Gunicorn
# ------------------------------------------------------------
# Adjust the module path (`src.main:app`) to match where your Flask
# application object lives.  The `-b 0.0.0.0:8005` binds the API,
# while `-b 0.0.0.0:8000` (optional) can be used for the metrics
# endpoint if you expose it separately.  Here we run a single
# Gunicorn worker because the engine is I/O‑bound; you can increase
# `--workers` if you ever need more concurrency.
CMD ["gunicorn", "-b", "0.0.0.0:8005", "src.main:app"]

#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Load Docker secrets (if any) into environment variables.
# Each secret file name becomes an uppercase variable.
# ------------------------------------------------------------
if [[ -d /run/secrets ]]; then
  for secret_file in /run/secrets/*; do
    var_name=$(basename "$secret_file" | tr '[:lower:]' '[:upper:]')
    export "$var_name"="$(cat "$secret_file")"
  done
fi

# ------------------------------------------------------------
# If a static .env file exists (non‑secret defaults), source it.
# ------------------------------------------------------------
if [[ -f /app/.env ]]; then
  set -a
  source /app/.env
  set +a
fi

# ------------------------------------------------------------
# Finally exec the command passed via CMD.
# ------------------------------------------------------------
exec "$@" 

# Dockerfile (existing)
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add the new libs
RUN pip install matplotlib seaborn

# after installing requirements
COPY migrations/2025-11-30_add_aggressive_pool.sql /docker-entrypoint-initdb.d/


FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY simulate_wr.py .
ENTRYPOINT ["python", "/app/simulate_wr.py"]

# Copy the trained LightGBM model (make sure the path matches the code)
COPY models/lightgbm_model.txt /app/models/lightgbm_model.txt

# (optional) also copy the JSON weight file for the linear fallback
COPY config/lever_weights.json /app/config/lever_weights.json

# Existing Dockerfile (citadel/trader)
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add guard code
COPY src/guards/ ./src/guards/

# -------------------------------------------------
# Health‑check – expects the bot to expose /health returning {"status":"ok"}
# -------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fSs http://localhost:8000/health || exit 1

[project.scripts]
cqt-ingest = "cqt.data_ingest.collector:run"


# ---------- Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Copy the compiled wheels from the builder stage (if you keep the builder)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the whole source tree (including the new data_ingest package)
COPY src/ src/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the data directory (will be a volume in production)
RUN mkdir -p /app/data && chown -R 1000:1000 /app/data

# Expose the API port (your bot) – keep unchanged
EXPOSE 8000

# -----------------------------------------------------------------
# ENTRYPOINT – you can keep the bot as the main process and run the
# collector as a *background* process via supervisord or a simple &
# -----------------------------------------------------------------
# Example using a tiny supervisor (runs both processes)
RUN apt-get update && apt-get install -y dumb-init

# Use dumb-init as PID 1 to reap children cleanly
ENTRYPOINT ["dumb-init", "--"]

# The command starts the collector in the background and then the bot
CMD bash -c "python -u src/data_ingest/collector.py & exec python -u src/main.py"

 # Add the new broker SDKs (or just requests if you use REST)
RUN pip install --no-cache-dir \
    ib_insync \
    ctrader-sdk \
    ninjatrader-sdk \
    tradovate-api

FROM python:3.11-slim
WORKDIR /app
RUN pip install prometheus-api-client requests
COPY rollback_watcher.py .
CMD ["python", "rollback_watcher.py"]

# ------------------------------------------------------------
# 6️⃣  FastAPI service (runs alongside the engine)
# ------------------------------------------------------------
FROM python:3.12-slim AS api

WORKDIR /app

# Copy only the API‑related source (you can copy the whole src/ if you like)
COPY src/ src/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port (you already expose 8005 for the Flask API;
# we’ll reuse the same port for the FastAPI control plane)
EXPOSE 8005

# Entrypoint – run uvicorn with the FastAPI app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8005"]

version: "3.9"

services:
  engine:
    # Image built from the combined Dockerfile (engine + FastAPI)
    image: ghcr.io/${GITHUB_REPOSITORY_OWNER:-yourorg}/cqt-engine:${IMAGE_TAG:-latest}
    container_name: cqt-engine
    restart: unless-stopped
    env_file: [.env]                     # static defaults (non‑secret)
    secrets:
      - POSTGRES_PASSWORD
      - CQT_API_TOKEN
      - MT5_PASSWORD
      - IBKR_API_KEY
      - IBKR_SECRET
    ports:
      - "8005:8005"                     # **Both** Flask API (/healthz, /config…) **and** FastAPI control API
    networks:
      - cqt-net
