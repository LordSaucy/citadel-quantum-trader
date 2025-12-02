import os
from flask import Flask, jsonify
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from news_overlay import is_recent_high_impact

from .config import logger, env
from .risk_management import RiskManagementLayer
from .ultimate_confluence_system import bp as confluence_bp, DEFAULT_WEIGHTS
from .advanced_execution_engine import AdvancedExecutionEngine
from config_loader import Config
import threading, time, os

# ----------------------------------------------------------------------
# Flask app factory
# ----------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)

    # ------------------------------------------------------------------
    # CORS – needed because Grafana Text panel runs in the browser
    # ------------------------------------------------------------------
    from flask_cors import CORS
    CORS(app, origins=["https://grafana.cqt.example.com"])

    # ------------------------------------------------------------------
    # Register the ConfluenceController blueprint (handles /config/*)
    # ------------------------------------------------------------------
    app.register_blueprint(confluence_bp)

    # ------------------------------------------------------------------
    # Global objects – one per process (good enough for a single‑node engine)
    # ------------------------------------------------------------------
    risk_engine = RiskManagementLayer()
    confluence_system = type("DummyConfluence", (), {"get_current_score": lambda _: 85})()
    exec_engine = AdvancedExecutionEngine(risk_engine, confluence_system)

    # ------------------------------------------------------------------
    # Health endpoint – used by the Load Balancer
    # ------------------------------------------------------------------
    @app.route("/healthz", methods=["GET"])
    def health():
        return "OK", 200

    # ------------------------------------------------------------------
    # Simple metrics endpoint – Prometheus scrapes this
    # ------------------------------------------------------------------
    @app.route("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    # ------------------------------------------------------------------
    # Demo endpoint – place a trade (used by the validation script)
    # ------------------------------------------------------------------
       # ------------------------------------------------------------------
    # Demo endpoint – place a trade (used by the validation script)
    # ------------------------------------------------------------------
    @app.route("/simulate", methods=["POST"])
    def simulate():
        """
        Very small demo: accept JSON with symbol/side/price,
        run the risk‑engine + confluence checks, and pretend to send an order.
        The payload should contain:
            {
                "symbol": "EURUSD",
                "direction": "BUY" | "SELL",
                "entry_price": 1.0800,
                "qty": 0.01               # optional – defaults to 0.01
            }
        """
        from flask import request

        try:
            payload = request.get_json(force=True)
            symbol = payload["symbol"]
            side = payload["direction"].upper()
            price = float(payload["entry_price"])
            qty = float(payload.get("qty", 0.01))
        except Exception as exc:  # pragma: no cover
            logger.error("Invalid simulate payload: %s", exc)
            return jsonify({"error": "invalid payload"}), 400

        # ------------------------------------------------------------------
        # In a real deployment you would pull these values from the DB /
        # market data feed. For the demo we use placeholder numbers.
        # ------------------------------------------------------------------
        current_open_positions = 0          # pretend we have none open
        equity = 100_000.0                  # fake account equity
        high_water_mark = 105_000.0         # fake HWM (5 % above equity)

        # ------------------------------------------------------------------
        # Run the execution engine
        # ------------------------------------------------------------------
        result = exec_engine.execute_trade(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            current_open_positions=current_open_positions,
            equity=equity,
            high_water_mark=high_water_mark,
        )

        # ------------------------------------------------------------------
        # Return a JSON payload that the validation script can inspect
        # ------------------------------------------------------------------
        return jsonify(result)

    # ------------------------------------------------------------------
    # Optional: expose the current kill‑switch status (useful for Grafana alerts)
    # ------------------------------------------------------------------
    @app.route("/killswitch", methods=["GET"])
    def killswitch():
        active = risk_engine.kill_switch_active
        reason = risk_engine.kill_reason
        return jsonify({"active": active, "reason": reason or ""})

    return app


# ----------------------------------------------------------------------
# Application entry‑point – used by Docker (via entrypoint.sh) and by
# `python -m src.main` during local development / testing.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Respect the optional PORT env var (useful when running locally)
    port = int(env("PORT", 8005, int))
    host = env("HOST", "0.0.0.0")
    logger.info("Starting CQT Flask API on %s:%s", host, port)
    create_app().run(host=host, port=port, debug=False)


def start_config_watcher():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    last_mtime = os.path.getmtime(cfg_path)

    def watcher():
        nonlocal last_mtime
        while True:
            time.sleep(5)
            try:
                mtime = os.path.getmtime(cfg_path)
                if mtime != last_mtime:
                    Config()._load()
                    last_mtime = mtime
                    print("[CONFIG] Reloaded from config.yaml")
            except Exception as e:
                print("[CONFIG] Watcher error:", e)

    threading.Thread(target=watcher, daemon=True).start()

if __name__ == "__main__":
    start_config_watcher()
    # … start the bot …

from .shutdown import register_graceful_shutdown
import logging

log = logging.getLogger("citadel.main")

def main():
    register_graceful_shutdown()   # <‑‑ add this line **before** you start any threads / async loops
    log.info("🚀 Citadel Quantum Trader starting …")
    # … existing initialization (DB, broker connection, scheduler, etc.) …
    try:
        # Your existing run‑loop (could be asyncio.run(main_async()))
        run_bot()
    except Exception as exc:
        log.exception("💥 Unhandled exception – shutting down")
        # Optionally write a final checkpoint here as well
        raise
def load_last_checkpoint(session):
    row = session.execute(
        "SELECT payload FROM bot_checkpoint ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    data = row[0]   # JSONB column
    # Re‑insert positions & pending orders into the DB tables
    for pos in data["positions"]:
        session.merge(Position(**pos))   # upsert
    for po in data["pending_orders"]:
        session.merge(PendingOrder(**po))
    # Restore risk schedule if you store it in BotState
    if data.get("risk_schedule"):
        state = session.query(BotState).first()
        state.risk_schedule_json = json.dumps(data["risk_schedule"])
    session.commit()
    log.info("✅ Restored %d positions and %d pending orders from checkpoint",
             len(data["positions"]), len(data["pending_orders"]))

# In main():
register_graceful_shutdown()
with get_session() as session:
    load_last_checkpoint(session)
# then start the normal bot loop

def generate_signal(bar, previous_bar):
    # ... volatility breakout already passed ...

    # Convert bar timestamp to epoch seconds (assuming bar['timestamp'] is datetime)
    signal_ts = bar['timestamp'].timestamp()

    if is_recent_high_impact(signal_ts):
        logger.info("High‑impact news arrived <30 s before signal – discarding")
        return None

    # Continue with regime‑forecast, etc.
    ...

def build_broker_adapter():
    paper_mode = os.getenv("PAPER_MODE", "false").lower() == "true"
    shadow_mode = os.getenv("SHADOW_MODE", "false").lower() == "true"

    # New env var that tells us which broker to use
    broker_type = os.getenv("BROKER_TYPE", "mt5").lower()   # defaults to MT5

    if broker_type == "mt5":
        from .mt5_adapter import MT5Adapter
        AdapterCls = MT5Adapter
    elif broker_type == "ibkr":
        from .ibkr_adapter import IBKRAdapter
        AdapterCls = IBKRAdapter
    elif broker_type == "binance_futures":
        from .binance_futures_adapter import BinanceFuturesAdapter
        AdapterCls = BinanceFuturesAdapter
    else:
        raise ValueError(f"Unsupported BROKER_TYPE={broker_type}")

    # Pass the appropriate mode flag
    if shadow_mode:
        log.info("🕶 Starting in SHADOW‑MODE – orders will be logged, not sent")
        return AdapterCls(paper_mode=False)   # shadow mode = real creds, no send
    elif paper_mode:
        log.info("🧪 Starting in PAPER‑TRADING mode – using demo credentials")
        return AdapterCls(paper_mode=True)
    else:
        log.info("🚀 Starting in LIVE‑MODE – real orders will be sent")
        return AdapterCls(paper_mode=False)


import asyncio
from tracing import tracer
from prometheus_client import Counter, Histogram

order_success = Counter("order_success_total", "Successful orders")
order_failure = Counter("order_failure_total", "Failed orders")
order_latency = Histogram(
    "order_latency_seconds",
    "Latency from signal to broker ACK",
    buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0],
)

async def process_one_signal(signal):
    # Whole processing of a single signal is a trace span
    with tracer.start_as_current_span("process_signal") as span:
        span.set_attribute("symbol", signal.symbol)
        span.set_attribute("direction", signal.direction)

        # 1️⃣ Depth / LIR guard (another nested span)
        with tracer.start_as_current_span("depth_guard"):
            if not depth_guard_ok(signal):
                order_failure.inc()
                span.set_attribute("guard", "depth_failed")
                return

        # 2️⃣ Risk calculation
        with tracer.start_as_current_span("risk_calc") as risk_span:
            stake = risk_manager.compute_stake(signal.bucket_id, signal.equity)
            risk_span.set_attribute("risk_fraction", stake / signal.equity)

        # 3️⃣ Send order (measure latency)
        start = asyncio.get_event_loop().time()
        with tracer.start_as_current_span("send_order") as exec_span:
            result = await execution_engine.send_order(
                symbol=signal.symbol,
                volume=stake,
                direction=signal.direction,
                sl=signal.stop_loss,
                tp=signal.take_profit,
            )
        elapsed = asyncio.get_event_loop().time() - start
        order_latency.observe(elapsed)

        if result["retcode"] == 0:
            order_success.inc()
            span.set_attribute("order_status", "filled")
        else:
            order_failure.inc()
            span.set_attribute("order_status", "rejected")
            span.set_attribute("reject_code", result["retcode"])

        # 4️⃣ Record to ledger (final nested span)
        with tracer.start_as_current_span("ledger_write"):
            ledger.record_trade(...)

async def main_loop():
    while True:
        signal = await market_feed.wait_for_signal()
        asyncio.create_task(process_one_signal(signal))
        await asyncio.sleep(0)   # let the event loop schedule other tasks


signal_engine = SignalEngine()

while True:
    df = market_feed.get_latest_dataframe()   # 1‑minute candles, depth, volume, etc.
    regime = regime_detector.predict(df)      # returns "trend", "range", or "high_vol"
    confluence = signal_engine.score(df, regime_label=regime)

