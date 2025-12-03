FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The container runs as a non‑root user (same UID as the rest of the stack)
RUN useradd -u 1000 -m appuser
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
