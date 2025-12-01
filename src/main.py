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
    @app.route("/simulate", methods=["POST"])
    def simulate():
        """
        Very small demo: accept JSON with symbol/side/price,
        run the risk‑engine + confluence checks, and pretend to send an order.
        """
        from flask import request

        data = request.get_json(force=True)
        symbol = data["symbol
