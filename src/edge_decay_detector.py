# src/edge_decay_detector.py
import threading
import time
import datetime
import logging
from sqlalchemy import select, func
from prometheus_client import Counter, Gauge
import requests
import os

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
