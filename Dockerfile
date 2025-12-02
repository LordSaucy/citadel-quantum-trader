# ------------------------------------------------------------
# 1️⃣  Builder stage – compile / install everything we need
# ------------------------------------------------------------
FROM python:3.12-slim AS builder

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


 
