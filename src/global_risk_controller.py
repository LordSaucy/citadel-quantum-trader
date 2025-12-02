import time
import psycopg2
import os
from config_loader import Config
from prometheus_client import Gauge, start_http_server

cfg = Config().settings
DB_URI = cfg["db_uri"]
MAX_GLOBAL_RISK = 0.05   # 5 % of total equity

# Prometheus metric for visibility
global_risk_pct = Gauge("global_risk_percentage", "Current global risk as % of AUM")

def fetch_all_bucket_states(conn):
    """Return list of (bucket_id, equity, risk_fraction)"""
    cur = conn.cursor()
    cur.execute("""
        SELECT bucket_id,
               SUM(pnl) + %s AS equity,   -- start equity per bucket is %s (passed in config)
               risk_fraction
        FROM trades
        GROUP BY bucket_id, risk_fraction;
    """, (cfg["bucket_start_equity"], cfg["bucket_start_equity"]))
    return cur.fetchall()

def adjust_risk_fractions(conn, scaling_factor):
    """Scale down every bucket's risk_fraction by scaling_factor (0‑1)."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE bucket_meta
        SET risk_fraction = GREATEST(risk_fraction * %s, %s);
    """, (scaling_factor, cfg["min_risk_fraction"]))
    conn.commit()

def monitor_loop():
    start_http_server(9100)   # expose Prometheus metrics
    while True:
        with psycopg2.connect(DB_URI) as conn:
            states = fetch_all_bucket_states(conn)
            total_equity = sum(eq for _, eq, _ in states)
            total_risk   = sum(eq * rf for _, eq, rf in states)
            current_risk_pct = total_risk / total_equity if total_equity else 0.0
            global_risk_pct.set(current_risk_pct * 100)

            if current_risk_pct > MAX_GLOBAL_RISK:
                # Reduce every bucket's risk by the excess proportion
                excess = current_risk_pct / MAX_GLOBAL_RISK
                scaling = 1.0 / excess   # e.g., if 0.08/0.05 => scaling = 0.625
                adjust_risk_fractions(conn, scaling)
                print(f"[{time.strftime('%X')}] GLOBAL RISK EXCEEDED – scaling risk fractions by {scaling:.3f}")

        time.sleep(30)   # run every 30 s

if __name__ == "__main__":
    monitor_loop()

