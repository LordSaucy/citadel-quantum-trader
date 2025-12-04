# -*- coding: utf-8 -*-

Centralised Prometheus instrumentation for Citadel Quantum Trader.

* Gauges – current values (win‑rate, LIR, depth, kill‑switch, …)
* Counters – ever‑increasing totals (orders, rejections, guard hits, …)
* Histograms – latency distributions.
* Helper to start the HTTP endpoint.
* Small “client” helpers that query the Prometheus HTTP API (optional).

All metrics are deliberately **labelled** (bucket_id, symbol, asset_class, …) so
they can be sliced in Grafana without exploding cardinality.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ----------------------------------------------------------------------
# 1️⃣  Metric definitions
# ----------------------------------------------------------------------
# ---- Core trading metrics ------------------------------------------------
order_total = Counter(
    "cqt_orders_total",
    "Total number of orders attempted",
    ["bucket_id", "success"],  # success = "true" / "false"
)

order_success_total = Counter(
    "cqt_order_success_total",
    "Number of successful orders",
    ["bucket_id"],
)

order_latency_seconds = Histogram(
    "cqt_order_latency_seconds",
    "Latency from signal generation to order submission (seconds)",
    ["bucket_id"],
)

# ---- Performance / risk gauges -----------------------------------------
bucket_winrate = Gauge(
    "cqt_bucket_winrate",
    "Rolling win‑rate (last 200 trades) per bucket",
    ["bucket_id"],
)

drawdown_pct = Gauge(
    "cqt_drawdown_pct",
    "Current draw‑down as a fraction of equity (negative number)",
)

global_risk_percentage = Gauge(
    "cqt_global_risk_percentage",
    "Aggregate risk exposure as % of total AUM",
)

bucket_current_risk_fraction = Gauge(
    "cqt_bucket_current_risk_fraction",
    "Current risk‑fraction (0‑1) used for the next trade",
    ["bucket_id"],
)

# ---- Guard / safety metrics --------------------------------------------
arb_guard_hits = Counter(
    "cqt_arb_guard_hits_total",
    "Number of times an arb was rejected by any execution‑risk guard",
)

arb_successes = Counter(
    "cqt_arb_success_total",
    "Number of successful triangular arbitrage executions",
)

trade_skipped_total = Counter(
    "cqt_trade_skipped_total",
    "Number of trades skipped by pre‑trade guard",
    ["reason"],  # e.g. depth, latency, spread, volatility, lir, etc.
)

depth_guard_rejections = Counter(
    "cqt_depth_guard_rejections_total",
    "Number of trades rejected by the depth‑guard",
    ["bucket_id", "symbol"],
)

# ---- Liquidity / market‑depth gauges ------------------------------------
lir_gauge = Gauge(
    "cqt_liquidity_imbalance_ratio",
    "Liquidity Imbalance Ratio (LIR) per bucket and symbol",
    ["bucket_id", "symbol"],
)

depth_gauge = Gauge(
    "cqt_total_market_depth",
    "Sum of bid + ask volume for the top N DOM levels (units of contract)",
    ["bucket_id", "symbol"],
)

# Optional per‑price‑bucket depth (useful for heat‑maps)
depth_price_gauge = Gauge(
    "cqt_market_depth_price_bucket",
    "Depth per price bucket (bid or ask)",
    ["bucket_id", "symbol", "price", "side"],  # side = "bid" / "ask"
)

# ---- Guard diagnostic gauges -------------------------------------------
guard_latency_seconds = Gauge(
    "cqt_guard_latency_seconds",
    "Measured round‑trip latency to broker (seconds)",
)

guard_spread_pips = Gauge(
    "cqt_guard_spread_pips",
    "Current mid‑price spread in pips",
)

guard_atr = Gauge(
    "cqt_guard_atr",
    "Current ATR (absolute) used for volatility‑spike guard",
)

guard_lir = Gauge(
    "cqt_guard_liquidity_imbalance_ratio",
    "Liquidity‑Imbalance Ratio for the current signal",
)

# ---- Regime‑matching counters ------------------------------------------
regime_match_total = Counter(
    "cqt_regime_match_total",
    "Number of times a live regime matched a historical cluster",
)

regime_mismatch_total = Counter(
    "cqt_regime_mismatch_total",
    "Number of times a live regime failed to match any allowed cluster",
)

# ---- Kill‑switch -------------------------------------------------------
kill_switch_active = Gauge(
    "cqt_kill_switch_active",
    "1 when the kill‑switch is engaged, 0 otherwise",
)

# ---- Order throughput ---------------------------------------------------
orders_per_sec = Gauge(
    "cqt_orders_per_sec",
    "Number of orders processed per second (aggregate across all buckets)",
)

# ---- Data‑ingest freshness ---------------------------------------------
cqt_latest_data_age_seconds = Gauge(
    "cqt_latest_data_age_seconds",
    "Age of the most recent OHLCV file per symbol (seconds)",
    ["symbol"],
)

cqt_last_ingest_timestamp = Gauge(
    "cqt_last_ingest_timestamp",
    "Unix timestamp of the most recent successful data ingest",
)

# ---- Asset‑class aware latency / reject / slippage --------------------
cqt_latency_seconds = Gauge(
    "cqt_latency_seconds",
    "Order submission latency (seconds)",
    ["symbol", "asset_class"],
)

cqt_reject_total = Counter(
    "cqt_reject_total",
    "Total number of order rejections",
    ["symbol", "asset_class"],
)

cqt_slippage = Gauge(
    "cqt_slippage",
    "Absolute slippage per filled order (pips or USD)",
    ["symbol", "asset_class"],
)

# ----------------------------------------------------------------------
# 2️⃣  Helper functions to update metrics
# ----------------------------------------------------------------------
def update_winrate(bucket_id: int, winrate: float) -> None:
    """Set the rolling win‑rate for a bucket."""
    bucket_winrate.labels(bucket_id=str(bucket_id)).set(winrate)


def set_bucket_risk(bucket_id: int, fraction: float) -> None:
    """Store the risk‑fraction (0‑1) that will be used for the next trade."""
    bucket_current_risk_fraction.labels(bucket_id=str(bucket_id)).set(fraction)


def record_order_success(bucket_id: int) -> None:
    """Increment counters & histograms after a successful order."""
    order_success_total.labels(bucket_id=str(bucket_id)).inc()
    order_total.labels(bucket_id=str(bucket_id), success="true").inc()


def record_order_failure(bucket_id: int) -> None:
    """Increment failure counters."""
    order_total.labels(bucket_id=str(bucket_id), success="false").inc()


def record_order_latency(bucket_id: int, secs: float) -> None:
    """Observe order latency in the histogram."""
    order_latency_seconds.labels(bucket_id=str(bucket_id)).observe(secs)


def record_guard_hit(reason: str) -> None:
    """Generic guard‑hit counter (e.g. depth, volatility, spread)."""
    trade_skipped_total.labels(reason=reason).inc()


def record_lir(bucket_id: int, symbol: str, lir: float) -> None:
    """Publish the Liquidity‑Imbalance Ratio for a bucket / symbol."""
    lir_gauge.labels(bucket_id=str(bucket_id), symbol=symbol).set(lir)


def record_market_depth(bucket_id: int, symbol: str, depth: float) -> None:
    """Publish total market depth (bid + ask) for a bucket / symbol."""
    depth_gauge.labels(bucket_id=str(bucket_id), symbol=symbol).set(depth)


def record_depth_price(
    bucket_id: int,
    symbol: str,
    price: float,
    side: str,
    volume: float,
) -> None:
    """
    Record depth per price bucket – useful for heat‑maps.

    Parameters
    ----------
    side: "bid" or "ask"
    """
    depth_price_gauge.labels(
        bucket_id=str(bucket_id),
        symbol=symbol,
        price=f"{price:.5f}",
        side=side,
    ).set(volume)


def trigger_kill_switch() -> None:
    """Engage the global kill‑switch (sets gauge to 1)."""
    kill_switch_active.set(1)


def clear_kill_switch() -> None:
    """Disengage the kill‑switch (sets gauge back to 0)."""
    kill_switch_active.set(0)


def inc_orders_per_sec(increment: float = 1.0) -> None:
    """Increment the aggregate orders‑per‑second gauge."""
    orders_per_sec.inc(increment)


def update_data_age(symbol: str, age_seconds: float) -> None:
    """Update the age of the newest OHLCV file for a given symbol."""
    cqt_latest_data_age_seconds.labels(symbol=symbol).set(age_seconds)


def set_last_ingest_timestamp(ts: int) -> None:
    """Set the Unix timestamp of the most recent successful ingest."""
    cqt_last_ingest_timestamp.set(ts)


def record_latency(symbol: str, asset_class: str, secs: float) -> None:
    """Record latency for a specific symbol / asset class."""
    cqt_latency_seconds.labels(symbol=symbol, asset_class=asset_class).set(secs)


def record_reject(symbol: str, asset_class: str) -> None:
    """Increment rejection counter for a specific symbol / asset class."""
    cqt_reject_total.labels(symbol=symbol, asset_class=asset_class).inc()


def record_slippage(symbol: str, asset_class: str, amount: float) -> None:
    """Set slippage (pips or USD) for a specific symbol / asset class."""
    cqt_slippage.labels(symbol=symbol, asset_class=asset_class).set(amount)


def record_regime_match(matched: bool) -> None:
    """Increment the appropriate regime‑match counter."""
    if matched:
        regime_match_total.inc()
    else:
        regime_mismatch_total.inc()


# ----------------------------------------------------------------------
# 3️⃣  Server starter
# ----------------------------------------------------------------------
def start_prometheus(port: int = 9090) -> None:
    """
    Start the Prometheus HTTP endpoint in a background thread.

    Call this **once** at process start, before any metric is emitted.
    """
    start_http_server(port)


# ----------------------------------------------------------------------
# 4️⃣  Optional lightweight Prometheus‑query client
# ----------------------------------------------------------------------
_PROM_ENDPOINT = "http://localhost:9090/api/v1/query"


def _query(expr: str) -> Optional[float]:
    """
    Execute a Prometheus instant query and return the first sample value.

    Returns ``None`` if the query fails or yields no results.
    """
    try:
        resp = httpx.get(_PROM_ENDPOINT, params={"query": expr}, timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            return None
        results = payload["data"]["result"]
        if not results:
            return None
        # result[0]["value"] = [timestamp, "<value_as_string>"]
        return float(results[0]["value"][1])
    except Exception:
        return None


def get_max_order_latency() -> float:
    """Maximum observed order‑submission latency (seconds)."""
    return _query("max(cqt_order_latency_seconds)") or 0.0


def get_reject_rate_per_sec() -> float:
    """
    Reject rate per second, averaged over the last 5 minutes.

    Multiply by 3600 to obtain a per‑hour rate, or by 100 for a percent.
    """
    return _query("rate(cqt_reject_total[5m])") or 0.0


def get_average_slippage() -> float:
    """Mean slippage across all filled orders (pips)."""
    return _query("avg(cqt_slippage)") or 0.0


def get_current_drawdown() -> float:
    """Current draw‑down as a negative fraction (e.g. -0.12)."""
    return _query("cqt_drawdown_pct") or 0.0


def get_overall_win_rate() -> float:
    """Overall win‑rate (fraction between 0 and 1)."""
    return _query(
        "sum(cqt_order_success_total) / sum(cqt_orders_total)"
    ) or 0.0


# ----------------------------------------------------------------------
# 5️⃣  Example usage (keep this at the bottom – it will not run on import)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Start the metrics endpoint on the default port (9090)
    start_prometheus()

    # Emit a few demo metrics so you can see them in Grafana / Prometheus UI
    update_winrate(bucket_id=1, winrate=0.62)
    set_bucket_risk(bucket_id=1, fraction=0.15)
    record_order_success(bucket_id=1)
    record_order_latency(bucket_id=1, secs=0.128)
    record_lir(bucket_id=1, symbol="EURUSD", lir=0.12)
    record_market_depth(bucket_id=1, symbol="EURUSD", depth=250_000)
    trigger_kill_switch()
    time.sleep(2)
    clear_kill_switch()
    print("Prometheus metrics are now exposed on http://localhost:9090/metrics")
