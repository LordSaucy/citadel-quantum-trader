# -------------------------------------------------
# Existing builder stage (unchanged)
# -------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# -------------------------------------------------
# Runtime stage for the *trading bot* (unchanged)
# -------------------------------------------------
FROM python:3.11-slim AS bot
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["python", "src/main.py"]

# -------------------------------------------------
# NEW stage – report runner (lightweight)
# -------------------------------------------------
FROM python:3.11-slim AS reporter
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY scripts/generate_report.py /app/
# Install only the extra packages needed for reporting (if you kept them in a separate file)
COPY requirements-report.txt /app/
RUN pip install -r requirements-report.txt

# The entrypoint will be overridden by systemd; we just need the image built.
