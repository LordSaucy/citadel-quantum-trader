from prometheus_api_client import PrometheusConnect
from datetime import datetime, timezone

PROM_URL = "http://prometheus:9090"
pc = PrometheusConnect(url=PROM_URL, disable_ssl=True)


def query_drawdown(start: str = None, end: str = None):
    """
    Returns a list of (timestamp, drawdown_pct) points.
    If start/end omitted, defaults to last 7 days.
    """
    if not start:
        start = (datetime.datetime.utcnow(datetime.timezone.utc) - datetime.timedelta(days=7)).isoformat()
    if not end:
        end = datetime.datetime.utcnow().isoformat()

    result = pc.custom_query_range(
        query="drawdown_pct",
        start_time=start,
        end_time=end,
        step="60s",  # 1‑minute resolution – adjust as needed
    )
    # Result format: [{'metric': {...}, 'values': [[ts, "0.123"], ...]}, ...]
    points = result[0]["values"] if result else []
    # Convert epoch seconds → ISO string for the front‑end
    return [{"ts": datetime.datetime.utcfromtimestamp(v[0]).isoformat() + "Z",
             "value": float(v[1])} for v in points]
