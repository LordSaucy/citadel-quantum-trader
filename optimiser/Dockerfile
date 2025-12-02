# ---- Builder ----
FROM python:3.11-slim AS builder
WORKDIR /opt/optimiser
COPY optimiser/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Runtime ----
FROM python:3.11-slim
WORKDIR /opt/optimiser
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY optimiser/ .          # includes run_opt.py, config.yaml (template), etc.
ENV PYTHONUNBUFFERED=1
# Entrypoint runs the optimiser in a loop (default: once per day)
CMD ["python", "run_opt.py"]
