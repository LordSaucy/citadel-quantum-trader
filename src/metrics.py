from prometheus_client import Gauge
from prometheus_client import Counter, Gauge


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
