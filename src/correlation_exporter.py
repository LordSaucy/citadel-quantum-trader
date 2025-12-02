# src/correlation_exporter.py
from prometheus_client import start_http_server, Gauge
import time
from sqlalchemy import create_engine, text

engine = create_engine("<your_db_uri>")
g_avg_corr = Gauge("cqt_average_correlation", "30‑day rolling average correlation across the basket")

def fetch_latest():
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT avg_corr FROM correlation_snapshots
            ORDER BY ts DESC LIMIT 1;
        """)).fetchone()
        return float(res[0]) if res else 0.0

if __name__ == "__main__":
    start_http_server(9101)          # expose on :9101/metrics
    while True:
        g_avg_corr.set(fetch_latest())
        time.sleep(60)               # update once per minute
