FROM python:3.11-slim AS builder
WORKDIR /app

# System deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY edge_decay_detector.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app
ENV PYTHONUNBUFFERED=1
CMD ["python", "edge_decay_detector.py"]
