FROM python:3.11-slim AS builder
WORKDIR /app

# Install system deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY src/global_risk_controller.py .
COPY src/config_loader.py .
COPY requirements.txt .   # make sure it contains psycopg2-binary, prometheus_client, pyyaml, hvac (if you use Vault)

RUN pip install --no-cache-dir -r requirements.txt

# Runtime (no build‑tools)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
CMD ["python", "global_risk_controller.py"]
