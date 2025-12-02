from prometheus_client import Gauge
from prometheus_client import Counter, Gauge
from src.edge_decay_detector import edge_decay_events_total, 

bucket_winrate = Gauge(
    "bucket_winrate",
    "Rolling win‑rate (last 200 trades) per bucket",
    ["bucket_id"]
)

def update_winrate(bucket_id: int, winrate: float):
    bucket_winrate.labels(bucket_id=str(bucket_id)).set(winrate)

arb_guard_hits = Counter(
    "arb_guard_hits_total",
    "Number of times an arb was rejected by any execution‑risk guard",
)

arb_successes = Counter(
    "arb_success_total",
    "Number of successful triangular arbitrage executions",
)

order_success_total = Counter("order_success_total", "Successful orders")
order_total = Counter("order_total", "All orders")
order_latency_seconds = Histogram("order_latency_seconds", "Latency per order")
drawdown_pct = Gauge("drawdown_pct", "Current draw‑down as fraction")
global_risk_percentage = Gauge("global_risk_percentage",
                               "Aggregate risk exposure as % of total AUM")

edge_decay_current_wr
