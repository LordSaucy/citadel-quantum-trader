# src/edge_decay_detector.py
import threading
import time
import datetime
import logging
from sqlalchemy import select, func
from prometheus_client import Counter, Gauge
import requests
import os
from prometheus_api_client import PrometheusConnect


log = logging.getLogger("edge_decay")
log.setLevel(logging.INFO)

# -------------------------------------------------
# Prometheus metrics
# -------------------------------------------------
edge_decay_events_total = Counter(
    "edge_decay_events_total",
    "Number of times edge‑decay detector tightened risk"
)


# Optional gauge to expose the *current* win‑rate (for Grafana)
edge_decay_current_wr = Gauge(
    "edge_decay_current_winrate",
    "Rolling 200‑trade win‑rate used by edge‑decay detector"
)

WIN_RATE_FLOOR = 0.95          # 95 % over the last 200 trades
RISK_MODIFIER = 0.5           # halve risk_fraction
MODIFIER_TRADES = 10          # apply for the next N trades

async def fetch_bucket_winrate(bucket_id: int, prometheus_url: str) -> float:
    """Query Prometheus for the rolling 200‑trade win‑rate of a bucket."""
    query = f"""
    sum(increase(wins_total{{bucket_id="{bucket_id}"}}[200]))
    /
    sum(increase(trades_total{{bucket_id="{bucket_id}"}}[200]))
    """
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{prometheus_url}/api/v1/query", params={"query": query}) as r:
            data = await r.json()
            try:
                return float(data["data"]["result"][0]["value"][1])
            except (IndexError, KeyError):
                return 1.0   # assume perfect if no data yet

async def apply_temporary_modifier(bucket_id: int, conn):
    """Write a temporary modifier row that expires after MODIFIER_TRADES."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO risk_modifier (bucket_id, multiplier, remaining_trades)
        VALUES (%s, %s, %s)
        ON CONFLICT (bucket_id) DO UPDATE
        SET multiplier = %s,
            remaining_trades = %s;
        """,
        (bucket_id, RISK_MODIFIER, MODIFIER_TRADES,
         RISK_MODIFIER, MODIFIER_TRADES)
    )
    conn.commit()

async def decay_loop(prometheus_url: str, db_uri: str):
    """Runs forever – checks every minute."""
    while True:
        async with aiohttp.ClientSession() as _:
            # Get list of bucket IDs from DB
            with psycopg2.connect(db_uri) as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT bucket_id FROM trades")
                bucket_ids = [row[0] for row in cur.fetchall()]

                for bid in bucket_ids:
                    winrate = await fetch_bucket_winrate(bid, prometheus_url)
                    if winrate < WIN_RATE_FLOOR:
                        # Edge‑decay triggered
                        edge_decay_events_total.inc()
                        apply_temporary_modifier(bid, conn)
                        print(f"[{time.strftime('%X')}] Edge‑decay: bucket {bid} win‑rate {winrate:.2%} → risk halved")
        await asyncio.sleep(60)   # run once per minute

if __name__ == "__main__":
    # Pull values from environment (same as other services)
    PROM_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    DB_URL   = os.getenv("DB_URI")
    asyncio.run(decay_loop(PROM_URL, DB_URL))

# -------------------------------------------------
# Configuration (environment variables – easy to tune)
# -------------------------------------------------
WIN_RATE_FLOOR = float(os.getenv("EDGE_DECAY_WINRATE_FLOOR", "0.95"))
WINDOW_SIZE = int(os.getenv("EDGE_DECAY_WINDOW_TRADES", "200"))
MODIFIER_MULTIPLIER = float(os.getenv("EDGE_DECAY_MULTIPLIER", "0.5"))
MODIFIER_TRADES = int(os.getenv("EDGE_DECAY_MODIFIER_TRADES", "10"))
SLACK_WEBHOOK = os.getenv("SLACK_EDGE_DECAY_WEBHOOK")  # optional

# -------------------------------------------------
# Helper: send a Slack message (if webhook configured)
# -------------------------------------------------
def _notify_slack(bucket_id: int, win_rate: float):
    if not SLACK_WEBHOOK:
        return
    payload = {
        "text": f":warning: *Edge‑Decay detected* – bucket {bucket_id} win‑rate "
                f"{win_rate:.2%} < floor {WIN_RATE_FLOOR:.2%}. "
                f"Risk fraction halved for next {MODIFIER_TRADES} trades."
    }
    try:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
    except Exception as exc:
        log.error("Failed to post to Slack: %s", exc)


# -------------------------------------------------
# Main detector class – runs in a background thread
# -------------------------------------------------
class EdgeDecayDetector(threading.Thread):
    def __init__(self, engine, risk_manager):
        super().__init__(daemon=True, name="EdgeDecayDetector")
        self.engine = engine
        self.risk_manager = risk_manager
        self.stop_event = threading.Event()

    def run(self):
        log.info("Edge‑Decay detector started (check every 60 s)")
        while not self.stop_event.wait(60):   # run once per minute
            try:
                self._process_all_buckets()
            except Exception as exc:
                log.exception("Unhandled error in Edge‑Decay detector: %s", exc)

    def stop(self):
        self.stop_event.set()

    # -------------------------------------------------
    # Process every bucket that has trades
    # -------------------------------------------------
    def _process_all_buckets(self):
        with self.engine.begin() as conn:
            # Get distinct bucket IDs that have at least WINDOW_SIZE trades
            bucket_ids = conn.execute(
                select(trades_tbl.c.bucket_id)
                .group_by(trades_tbl.c.bucket_id)
                .having(func.count(trades_tbl.c.id) >= WINDOW_SIZE)
            ).scalars().all()

        for bucket_id in bucket_ids:
            win_rate = self._calc_win_rate(bucket_id)
            edge_decay_current_wr.set(win_rate)   # expose to Prometheus

            if win_rate < WIN_RATE_FLOOR:
                # Apply the temporary modifier
                self.risk_manager.apply_temporary_modifier(
                    bucket_id=bucket_id,
                    multiplier=MODIFIER_MULTIPLIER,
                    trades=MODIFIER_TRADES,
                )
                edge_decay_events_total.inc()
                log.info(
                    "Edge‑decay triggered for bucket %s: win‑rate %.2f%% < %.2f%%",
                    bucket_id, win_rate * 100, WIN_RATE_FLOOR * 100,
                )
                _notify_slack(bucket_id, win_rate)

    # -------------------------------------------------
    # Compute rolling 200‑trade win‑rate for a bucket
    # -------------------------------------------------
    def _calc_win_rate(self, bucket_id: int) -> float:
        with self.engine.begin() as conn:
            # Grab the most recent WINDOW_SIZE trades (ordered by timestamp)
            sub = (
                select(trades_tbl.c.pnl)
                .where(trades_tbl.c.bucket_id == bucket_id)
                .order_by(trades_tbl.c.timestamp.desc())
                .limit(WINDOW_SIZE)
                .subquery()
            )
            # Count wins (+) vs total
            wins = conn.execute(select(func.sum(func.case([(sub.c.pnl > 0, 1)], else_=0))).scalar()
            total = conn.execute(select(func.count(sub.c.pnl))).scalar()
            if total == 0:
                return 0.0
            return float(wins) / float(total)

            pc = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)

def high_drift_present() -> bool:
    # If any feature PSI > 0.25 in the last 5 minutes → True
    query = "max_over_time(feature_psi_*[5m]) > 0.25"
    result = pc.custom_query(query=query)
    return bool(result)   # non‑empty list means at least one feature crossed

def maybe_trigger_decay():
    if high_drift_present():
        # Immediately tighten the risk schedule (e.g., set all future
        # risk fractions to the minimum allowed value)
        log.warning("⚠️ High feature drift detected – tightening risk schedule")
        # You can call the same function that the win‑rate floor uses,
        # or directly set a global flag that the RiskManagementLayer reads.
        set_global_risk_multiplier(0.5)   # example: halve all risk fractions

