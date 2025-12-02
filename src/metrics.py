from prometheus_client import Gauge

bucket_winrate = Gauge(
    "bucket_winrate",
    "Rolling win‑rate (last 200 trades) per bucket",
    ["bucket_id"]
)

def update_winrate(bucket_id: int, winrate: float):
    bucket_winrate.labels(bucket_id=str(bucket_id)).set(winrate)
