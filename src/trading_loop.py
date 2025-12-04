from market_data.dom_cache import DomCache
from market_data.lir import compute_lir
from utils.depth_guard import depth_guard   # the function we wrote earlier
from risk_management.risk_manager import RiskManager
from execution_engine.execution_engine import ExecutionEngine
import logging

log = logging.getLogger(__name__)

# Assume you already have instances:
risk_manager   = RiskManager()
exec_engine    = ExecutionEngine()
dom_cache      = DomCache()   # singleton already running

def process_signal(signal: dict):
    """
    Called for every candidate signal produced by the strategy layer.
    `signal` must contain at least:
        - symbol
        - direction ("BUY"/"SELL")
        - entry_price (float)
        - bucket_id (risk bucket identifier)
    """
    symbol      = signal["symbol"]
    direction   = signal["direction"]
    entry_price = float(signal["entry_price"])
    bucket_id   = signal["bucket_id"]

    # -------------------------------------------------------------
    # 1️⃣  Risk‑manager approval (draw‑down, position limits, etc.)
    # -------------------------------------------------------------
    ok, reason = risk_manager.evaluate_signal(
        bucket_id=bucket_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
    )
    if not ok:
        log.info("Signal rejected by risk manager: %s", reason)
        return

    # -------------------------------------------------------------
    # 2️⃣  Pull the latest DOM from the cache (thread‑safe, no RPC)
    # -------------------------------------------------------------
    dom_df = dom_cache.get(symbol)

    # -------------------------------------------------------------
    # 3️⃣  Compute LIR (used later for scoring & for the gauge)
    # -------------------------------------------------------------
    lir = compute_lir(dom_df)

    # -------------------------------------------------------------
    # 4️⃣  Update Prometheus gauges (optional but highly recommended)
    # -------------------------------------------------------------
    # Assuming you have a Prometheus CollectorRegistry somewhere:
    from metrics.prometheus import lir_gauge, depth_gauge
    lir_gauge.labels(bucket_id=str(bucket_id), symbol=symbol).set(lir)
    total_depth = dom_df["bid_volume"].sum() + dom_df["ask_volume"].sum()
    depth_gauge.labels(bucket_id=str(bucket_id), symbol=symbol).set(total_depth)

    # -------------------------------------------------------------
    # 5️⃣  Apply the *depth guard* – reject if not enough volume
    # -------------------------------------------------------------
    equity = risk_manager.current_equity()
    stake_usd = risk_manager.compute_stake(bucket_id, equity)

    if not depth_guard(entry_price, dom_df, stake_usd):
        log.warning(
            "Depth guard rejected trade %s %s @ %.5f – insufficient volume",
            direction, symbol, entry_price
        )
        # Still deduct the risk‑fraction so win‑rate stays honest
        risk_manager.record_failed_trade(bucket_id, stake_usd)
        return

    # -------------------------------------------------------------
    # 6️⃣  All checks passed → send the order to the broker
    # -------------------------------------------------------------
    exec_engine.send_order(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stake_usd=stake_usd,
        # you can also pass the LIR as a “confluence” weight if you like:
        extra_context={"lir": lir}
    )

    # -------------------------------------------------------------
    # 7️⃣  Tell the risk manager that we actually placed a trade
    # -------------------------------------------------------------
    risk_manager.record_successful_trade(bucket_id, stake_usd)
    log.info(
        "Order SENT – %s %s @ %.5f (stake $%.2f, LIR %.3f)",
        direction, symbol, entry_price, stake_usd, lir
    )
