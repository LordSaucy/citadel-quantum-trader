# ==========================
# admin_ui/backend/Dockerfile
# ==========================

# ---------- Builder ----------
FROM python:3.11-slim AS builder

ARG BUILD_UID=1000
ARG BUILD_GID=1000
RUN addgroup --gid ${BUILD_GID} builder && \
    adduser --uid ${BUILD_UID} --gid ${BUILD_GID} --disabled-password --gecos "" builder

WORKDIR /app

# System packages needed to compile wheels (psycopg2, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (locked versions)
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Runtime ----------
FROM python:3.11-slim AS runtime

ARG APP_UID=1000
ARG APP_GID=1000
RUN addgroup --gid ${APP_GID} cqt && \
    adduser --uid ${APP_UID} --gid ${APP_GID} --disabled-password --gecos "" cqt

WORKDIR /app

# Runtime OS deps (only libpq runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# FastAPI source
COPY backend/app ./app

# Pre‑built Vite UI (produced by the multi‑stage Vite build)
COPY backend/app/static ./app/static

# Make sure the non‑root user owns everything
RUN chown -R cqt:cqt /app

# Expose FastAPI HTTP port
EXPOSE 8000

# Run FastAPI with uvicorn
USER cqt
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
