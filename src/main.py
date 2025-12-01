import os
from flask import Flask, jsonify
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .config import logger, env
from .risk_management import RiskManagementLayer
from .ultimate_confluence_system import bp as confluence_bp, DEFAULT_WEIGHTS
from .advanced_execution_engine import AdvancedExecutionEngine

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
