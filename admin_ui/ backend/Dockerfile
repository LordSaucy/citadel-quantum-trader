# ==============================
# 5.1 Backend Dockerfile (admin_ui/backend/Dockerfile)
# ==============================

# -----------------------------------------------------------------
#   👷‍♂️  Builder stage – install Python deps and compile wheels
# -----------------------------------------------------------------
FROM python:3.11-slim AS builder

# Create a non‑root user for the build (optional, but helps keep permissions sane)
ARG BUILD_UID=1000
ARG BUILD_GID=1000
RUN addgroup --gid $BUILD_GID builder && \
    adduser --uid $BUILD_UID --gid $BUILD_GID --disabled-password --gecos "" builder

WORKDIR /app

# Install system packages required for building any binary wheels (e.g. psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first – this layer is cached unless the file changes
COPY backend/requirements.txt .

# Upgrade pip and install the exact versions from the lock file
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------
#   🏃‍♀️  Runtime stage – copy the compiled deps and the app source
# -----------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create a dedicated non‑root user that will run the app
ARG APP_UID=1000
ARG APP_GID=1000
RUN addgroup --gid $APP_GID cqt && \
    adduser --uid $APP_UID --gid $APP_GID --disabled-password --gecos "" cqt

WORKDIR /app

# Install only the runtime system packages we actually need
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy the compiled Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the FastAPI source code
COPY backend/app ./app

# -------------------------------------------------------------
#   📁  Include the pre‑built frontend assets (Vite output)
# -------------------------------------------------------------
# The Vite build step places the static files in
#   admin_ui/backend/app/static
# (see the multi‑stage Dockerfile that builds the UI).  We simply
# copy that directory into the runtime image so FastAPI can serve it.
COPY backend/app/static ./app/static

# Ensure the non‑root user owns the application files
RUN chown -R cqt:cqt /app

# -------------------------------------------------------------
#   🌐  Expose the HTTP port used by FastAPI
# -------------------------------------------------------------
EXPOSE 8000

# -------------------------------------------------------------
#   🚀  Entrypoint – start the ASGI server
# -------------------------------------------------------------
# Use uvicorn with a single worker (you can increase workers if you
# expect high concurrency; 1 worker is fine for the admin UI).
USER cqt
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
