from prometheus_client import Gauge
from prometheus_client import Counter, Gauge
from src.edge_decay_detector import edge_decay_events_total, 
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Gauge, start_http_server

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

epth_ok_gauge = Gauge(
    "depth_check_ok_total",
    "Number of depth checks that passed the minimum‑depth rule",
    ["bucket_id", "symbol"]
)


# -------------------------------------------------
# New gauge – per‑bucket risk fraction (0‑1)
# -------------------------------------------------
bucket_current_risk_fraction = Gauge(
    "bucket_current_risk_fraction",
    "Current risk‑fraction (0‑1) used for the next trade",
    ["bucket_id"]
)

def set_bucket_risk(bucket_id: int, fraction: float):
    """Call this after compute_stake() decides the fraction."""
    bucket_current_risk_fraction.labels(bucket_id=str(bucket_id)).set(fraction)

# Count how many trades were prevented and why
trade_skipped_total = Counter(
    "trade_skipped_total",
    "Number of trades skipped by pre‑trade guard",
    ["reason"]  # possible values: depth, latency, spread, volatility
)

# Optional: expose the latest guard‑check values for debugging
guard_latency_seconds = Gauge("guard_latency_seconds",
                             "Measured round‑trip latency to broker (seconds)")
guard_spread_pips = Gauge("guard_spread_pips",
                         "Current mid‑price spread in pips")
guard_atr = Gauge("guard_atr",
                  "Current ATR (absolute) used for volatility‑spike guard")
guard_lir = Gauge("guard_liquidity_imbalance_ratio",
                  "Liquidity‑Imbalance Ratio (LIR) for the current signal")

regime_match_total = Counter(
    "regime_match_total",
    "Number of times a live regime matched a historical cluster"
)

regime_mismatch_total = Counter(
    "regime_mismatch_total",
    "Number of times a live regime failed to match any allowed cluster"
)

def record_regime_match(matched: bool):
    if matched:
        regime_match_total.inc()
    else:
        regime_mismatch_total.inc()
# LIR per bucket (signed float)
lir_gauge = Gauge(
    "bucket_liquidity_imbalance_ratio",
    "Liquidity Imbalance Ratio for the bucket ([-1,1])",
    ["bucket_id"]
)

# Total depth (sum of bid+ask volume) at the best 20 levels – useful for alerting
depth_gauge = Gauge(
    "bucket_market_depth",
    "Sum of bid+ask volume (units) for the top 20 DOM levels",
    ["bucket_id"]
)


kill_switch_active = Gauge(
    "kill_switch_active",
    "1 when the kill‑switch is engaged, 0 otherwise",
)

# When the watchdog detects a latency breach, set it:
def trigger_kill_switch():
    kill_switch_active.set(1)
    # … existing logic that pauses the bot …

orders_per_sec = Gauge(
    "citadel_orders_per_sec",
    "Number of orders processed per second (aggregate across all buckets)",
)

# Increment this gauge in the order‑submission path:
def record_order():
    # after a successful order:
    orders_per_sec.inc()

latest_file_age_seconds = Gauge(
    "cqt_latest_data_age_seconds",
    "Age of the most recent OHLCV file per symbol",
    ["symbol"]
)

def update_data_age_metrics():
    for sym in SYMBOLS:
        # find newest file for this symbol
        files = list((BASE_DATA_DIR / sym).glob(f"{sym}_*.parquet"))
        if not files:
            continue
        newest = max(files, key=lambda p: p.stat().st_mtime)
        age = time.time() - newest.stat().st_mtime
        latest_file_age_seconds.labels(symbol=sym).set(age)

# Existing gauges – add `asset_class` label
LATENCY_GAUGE = Gauge(
    "cqt_latency_seconds",
    "Order submission latency (seconds)",
    ["symbol", "asset_class"]
)

REJECT_COUNTER = Counter(
    "cqt_reject_total",
    "Total number of order rejections",
    ["symbol", "asset_class"]
)

SLIPPAGE_GAUGE = Gauge(
    "cqt_slippage",
    "Absolute slippage per filled order (pips or usd)",
    ["symbol", "asset_class"]
)

def record_latency(symbol: str, asset_class: str, secs: float):
    LATENCY_GAUGE.labels(symbol=symbol, asset_class=asset_class).set(secs)

def record_reject(symbol: str, asset_class: str):
    REJECT_COUNTER.labels(symbol=symbol, asset_class=asset_class).inc()

def record_slippage(symbol: str, asset_class: str, amount: float):
    SLIPPAGE_GAUGE.labels(symbol=symbol, asset_class=asset_class).set(amount)


INGEST_TS = Gauge("cqt_last_ingest_timestamp", "Unix timestamp of the most recent successful data‑ingest")
def run():
    # … after successful write …
    INGEST_TS.set(int(time.time()))

depth_guard_rejections = Counter(
    "cqt_depth_guard_rejections_total",
    "Number of trades rejected by the depth‑guard",
    ["bucket_id", "symbol"]
)

# Inside depth_guard(...)
if not enough_volume:
    depth_guard_rejections.labels(bucket_id=str(bucket_id), symbol=symbol).inc()
    return False

depth_price_gauge = Gauge(
    "cqt_market_depth_price_bucket",
    "Depth per price bucket (bid or ask)",
    ["bucket_id", "symbol", "price", "side"]   # side = "bid" or "ask"
)
for _, row in dom_df.iterrows():
    depth_price_gauge.labels(
        bucket_id=str(bucket_id),
        symbol=symbol,
        price=f"{row['price']:.5f}",
        side="bid" if row['bid_volume'] > 0 else "ask"
    ).set(row['bid_volume'] if row['bid_volume'] > 0 else row['ask_volume'])
 
