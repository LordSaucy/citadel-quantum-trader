#!/usr/bin/env python3
"""
Webhook Server

Receives trading signals from TradingView and executes trades.

Features:
- Flask REST API
- HMAC signature validation (for /webhook endpoint)
- Token authentication (for /webhook endpoint)
- Rate‑limiting per IP
- Signal processing (BASE / STACK)
- Status monitoring endpoints
- CSRF protection on internal admin endpoints
- Graceful shutdown

✅ SECURITY:
- /webhook: Uses token + HMAC signature validation (external TradingView calls)
- /pause, /resume, /force_stack: CSRF‑protected (internal admin endpoints)
- GET endpoints: Unprotected (read‑only, safe by design)
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import sys
import threading
import time
import hmac
import hashlib
import os
import secrets
from datetime import datetime, timedelta, date, time as dt_time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/webhook_server.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "webhook_server.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration (adjust the values before production)
# ----------------------------------------------------------------------
class WebhookConfig:
    """All configurable options for the webhook server."""

    # ----- Security ----------------------------------------------------
    WEBHOOK_TOKEN: str = "CHANGE_ME_TO_A_STRONG_RANDOM_TOKEN"
    WEBHOOK_SECRET: str = "CHANGE_ME_TO_A_STRONG_RANDOM_HMAC_SECRET"

    # ----- Server -------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    # ----- Feature toggles ---------------------------------------------
    REQUIRE_SIGNATURE: bool = True   # HMAC validation
    REQUIRE_TOKEN: bool = True       # Token auth

    # ----- Rate limiting -----------------------------------------------
    MAX_REQUESTS_PER_MINUTE: int = 60


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
app = Flask(__name__)
CORS(app)   # allow dashboard / external UI access

# ✅ SECURITY FIX: Secure SECRET_KEY management
def get_or_create_app_secret_key() -> str:
    """
    Retrieve ``SECRET_KEY`` from the environment, a persisted file,
    or generate a new cryptographically‑secure key (fallback).

    Priority:
    1️⃣  ``FLASK_WEBHOOK_SECRET_KEY`` env var (production)
    2️⃣  Persisted key file (generated once, reused across restarts)
    3️⃣  Generate a new key (fallback)
    """
    # 1️⃣  Environment variable
    env_key = os.getenv('FLASK_WEBHOOK_SECRET_KEY')
    if env_key:
        logger.info("📌 Webhook SECRET_KEY loaded from FLASK_WEBHOOK_SECRET_KEY environment variable")
        return env_key

    # 2️⃣  Persisted key file
    secret_file = Path("./config/webhook_secret_key")
    if secret_file.exists():
        try:
            with open(secret_file, 'r') as f:
                persisted_key = f.read().strip()
                if persisted_key and len(persisted_key) >= 32:
                    logger.info("📌 Webhook SECRET_KEY loaded from persisted secure key file")
                    return persisted_key
        except Exception as exc:
            logger.warning(f"⚠️ Could not read persisted Webhook SECRET_KEY: {exc}")

    # 3️⃣  Generate a new key
    new_key = secrets.token_urlsafe(32)

    # Persist for future restarts
    try:
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        with open(secret_file, 'w') as f:
            f.write(new_key)
        secret_file.chmod(0o600)
        # NOTE: This is the line SonarQube flagged – it was an f‑string with no
        # interpolation.  Replaced with a normal string.
        logger.info("✅ Generated and persisted new Webhook SECRET_KEY")
    except Exception as exc:
        logger.warning(f"⚠️ Could not persist Webhook SECRET_KEY: {exc}")

    return new_key


# ✅ SECURITY FIX: Set SECRET_KEY and enable CSRF for admin endpoints
app.config['SECRET_KEY'] = get_or_create_app_secret_key()
csrf = CSRFProtect(app)

# Disable CSRF for the webhook endpoint (uses HMAC + token instead)
csrf.exempt(lambda: request.path == '/webhook' and request.method == 'POST')


# ----------------------------------------------------------------------
# Global runtime state
# ----------------------------------------------------------------------
TRADING_PAUSED: bool = False
signal_history: List[Dict] = []          # last 100 signals
request_counts: Dict[str, List[datetime]] = {}   # IP → timestamps (rate‑limit)


# ----------------------------------------------------------------------
# Helper – security
# ----------------------------------------------------------------------
def verify_hmac_signature(payload: bytes, signature: str) -> bool:
    """Validate the HMAC‑SHA256 signature sent by TradingView."""
    expected = hmac.new(
        WebhookConfig.WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_token(token: str) -> bool:
    """Simple token check."""
    return token == WebhookConfig.WEBHOOK_TOKEN


def check_rate_limit(ip: str) -> bool:
    """Allow up to ``MAX_REQUESTS_PER_MINUTE`` requests per IP."""
    now = datetime.now()
    window_start = now - timedelta(minutes=1)

    timestamps = request_counts.get(ip, [])
    # Discard old entries
    timestamps = [t for t in timestamps if t > window_start]
    if len(timestamps) >= WebhookConfig.MAX_REQUESTS_PER_MINUTE:
        return False

    timestamps.append(now)
    request_counts[ip] = timestamps
    return True


# ----------------------------------------------------------------------
# Endpoint: /webhook  (POST)
# ----------------------------------------------------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Expected JSON payload (example):
    {
        "type": "BASE" | "STACK",
        "signal": "BUY" | "SELL",
        "symbol": "EURUSD",
        "entry": 1.1050,
        "sl": 1.1020,
        "atr": 0.0030,
        "confluence": 4,
        "timestamp": "2024-01-15T10:30:00Z"
    }

    Security:
    - Requires valid X‑Webhook‑Token header
    - Requires valid X‑Webhook‑Signature (HMAC‑SHA256)
    - Rate limited per IP
    """
    ip = request.remote_addr or "unknown"

    # ---- Rate limiting -------------------------------------------------
    if not check_rate_limit(ip):
        logger.warning(f"Rate limit exceeded for {ip}")
        abort(429, "Rate limit exceeded")

    # ---- Token authentication -----------------------------------------
    if WebhookConfig.REQUIRE_TOKEN:
        token = request.headers.get("X-Webhook-Token")
        if not token or not verify_token(token):
            logger.warning(f"Invalid token from {ip}")
            abort(401, "Invalid token")

    # ---- HMAC signature validation ------------------------------------
    if WebhookConfig.REQUIRE_SIGNATURE:
        signature = request.headers.get("X-Webhook-Signature")
        if not signature or not verify_hmac_signature(request.data, signature):
            logger.warning(f"Invalid HMAC signature from {ip}")
            abort(401, "Invalid signature")

    # ---- Parse JSON ----------------------------------------------------
    try:
        data = request.get_json(force=True)
    except Exception as exc:
        logger.error(f"JSON decode error from {ip}: {exc}")
        abort(400, "Invalid JSON")

    # ---- Basic field validation ----------------------------------------
    required = ["type", "signal", "symbol", "entry", "sl"]
    missing = [f for f in required if f not in data]
    if missing:
        logger.error(f"Missing fields {missing} from {ip}")
        return jsonify(
            {"status": "error", "message": f"Missing fields: {', '.join(missing)}"}
        ), 400

    # ---- Pause check ----------------------------------------------------
    if TRADING_PAUSED:
        logger.info("Trading paused – signal ignored")
        return jsonify(
            {"status": "paused", "message": "Trading is currently paused"}
        ), 200

    # ---- Store in history (keep last 100) ------------------------------
    signal_entry = {
        **data,
        "received_at": datetime.now().isoformat(),
        "ip": ip,
    }
    signal_history.append(signal_entry)
    if len(signal_history) > 100:
        signal_history.pop(0)

    # ---- Log nicely ----------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"📥 Webhook received from {ip}")
    logger.info(json.dumps(data, indent=2))
    logger.info("=" * 80)

    # ---- Dispatch processing in a background thread --------------------
    threading.Thread(
        target=_process_signal, args=(data,), daemon=True
    ).start()

    return jsonify(
        {"status": "received", "timestamp": datetime.now().isoformat(), "data": data}
    ), 200


# ----------------------------------------------------------------------
# Endpoint: /status  (GET) – read‑only, safe
# ----------------------------------------------------------------------
@app.route("/status", methods=["GET"])
def status():
    """Return a quick health/status snapshot."""
    try:
        import MetaTrader5 as mt5
        from unified_trading_bot import UnifiedTradingBot
        from position_stacking_manager import stacking_manager
    except Exception as exc:   # pragma: no cover
        logger.error(f"Status endpoint import error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500

    acct = mt5.account_info()
    stacking = stacking_manager.get_status()

    return jsonify(
        {
            "status": "online",
            "trading_paused": TRADING_PAUSED,
            "timestamp": datetime.now().isoformat(),
            "account": {
                "balance": acct.balance if acct else 0,
                "equity": acct.equity if acct else 0,
                "margin_free": acct.margin_free if acct else 0,
            },
            "stacking": stacking,
            "signals_received": len(signal_history),
            "last_signal": signal_history[-1] if signal_history else None,
        }
    ), 200


# ----------------------------------------------------------------------
# Endpoint: /pause  (POST) – CSRF protected admin endpoint
# ----------------------------------------------------------------------
@app.route("/pause", methods=["POST"])
@csrf.protect
def pause_trading():
    """Pause trading via API (requires CSRF token)."""
    global TRADING_PAUSED
    TRADING_PAUSED = True
    logger.info("⏸️ Trading PAUSED via API")
    return jsonify({"status": "paused"}), 200


# ----------------------------------------------------------------------
# Endpoint: /resume  (POST) – CSRF protected admin endpoint
# ----------------------------------------------------------------------
@app.route("/resume", methods=["POST"])
@csrf.protect
def resume_trading():
    """Resume trading via API (requires CSRF token)."""
    global TRADING_PAUSED
    TRADING_PAUSED = False
    logger.info("▶️ Trading RESUMED via API")
    return jsonify({"status": "resumed"}), 200


# ----------------------------------------------------------------------
# Endpoint: /force_stack/  (POST) – CSRF protected admin endpoint
# ----------------------------------------------------------------------
@app.route("/force_stack/", methods=["POST"])
@csrf.protect
def force_stack(symbol: str):
    """Manually force a STACK entry (requires CSRF token)."""
    try:
        payload = request.get_json(force=True)
        entry = float(payload["entry"])
        sl = float(payload["sl"])
    except Exception as exc:
        logger.error(f"Force stack payload error: {exc}")
        return jsonify({"status": "error", "message": "Invalid payload"}), 400

    logger.info(f"🔧 Manual STACK request for {symbol} @ {entry} SL {sl}")

    try:
        from position_stacking_manager import stacking_manager
        ticket = stacking_manager.place_stack_position(symbol, entry, sl)
    except Exception as exc:   # pragma: no cover
        logger.error(f"Stack placement error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500

    if ticket:
        return jsonify({"status": "success", "ticket": ticket, "symbol": symbol}), 200
    else:
        return jsonify({"status": "failed", "message": "Stack placement failed"}), 400


# ----------------------------------------------------------------------
# Endpoint: /history  (GET) – read‑only
# ----------------------------------------------------------------------
@app.route("/history", methods=["GET"])
def get_history():
    """Return recent signal history."""
    limit = int(request.args.get("limit", 50))
    return jsonify(
        {"signals": signal_history[-limit:], "total": len(signal_history)}
    ), 200


# ----------------------------------------------------------------------
# Endpoint: /health  (GET) – read‑only
# ----------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.1",
        }
    ), 200


# ----------------------------------------------------------------------
# Internal: signal processing
# ----------------------------------------------------------------------
def _process_signal(data: Dict) -> None:
    """Core routine that receives a validated signal and hands it off to the main trading engine."""
    try:
        signal_type = data.get("type")          # BASE or STACK
        direction = data.get("signal")          # BUY or SELL
        symbol = data.get("symbol")
        entry = float(data.get("entry"))
        sl = float(data.get("sl"))
        atr = float(data.get("atr", 0))
        confluence = int(data.get("confluence", 0))
        timestamp = data.get("timestamp", "")

        logger.info(
            f"Processing {signal_type} {direction} for {symbol} @ {entry:.5f}"
        )

        # Optional: ignore stale signals (>5 min old)
        if timestamp:
            try:
                sig_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age = (datetime.now() - sig_time).total_seconds()
                if age > 300:
                    logger.warning(f"Ignoring stale signal ({age:.0f}s old)")
                    return
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 1️⃣  Forward to the unified trading bot
        # ------------------------------------------------------------------
        from unified_trading_bot import UnifiedTradingBot

        bot = UnifiedTradingBot.get_instance()   # singleton used throughout CQT

        if signal_type == "BASE":
            # Full 7‑lever validation pipeline
            bot.handle_base_signal(
                symbol=symbol,
                direction=direction,
                entry=entry,
                stop_loss=sl,
                atr=atr,
                confluence=confluence,
                raw_timestamp=timestamp,
            )
        elif signal_type == "STACK":
            # Directly push a stack position (bypass the full 7‑lever flow)
            from position_stacking_manager import stacking_manager

            ticket = stacking_manager.place_stack_position(symbol, entry, sl)
            if ticket:
                logger.info(f"✅ Stack position opened – ticket #{ticket}")
            else:
                logger.error("❌ Failed to open stack position")
        else:
            logger.warning(f"Unknown signal type: {signal_type}")

    except Exception as exc:   # pragma: no cover
        logger.error(f"Error processing signal: {exc}", exc_info=True)


# ----------------------------------------------------------------------
# Server start / graceful shutdown helpers
# ----------------------------------------------------------------------
def _run_flask():
    """Internal helper to start Flask – used by the background thread."""
    host = WebhookConfig.HOST
    port = WebhookConfig.PORT
    logger.info("=" * 80)
    logger.info("🌐 WEBHOOK SERVER STARTING")
    logger.info("=" * 80)
    logger.info(f"Listening on {host}:{port}")
    logger.info(
        f"Token auth: {'ENABLED' if WebhookConfig.REQUIRE_TOKEN else 'DISABLED'}"
    )
    logger.info(
        f"HMAC signature: {'ENABLED' if WebhookConfig.REQUIRE_SIGNATURE else 'DISABLED'}"
    )
    logger.info("✅ CSRF protection enabled on admin endpoints (/pause, /resume, /force_stack)")
    logger.info("Endpoints:")
    logger.info(f"  POST   http://{host}:{port}/webhook              (token + HMAC auth)")
    logger.info(f"  GET    http://{host}:{port}/status")
    logger.info(f"  POST   http://{host}:{port}/pause               (CSRF protected)")
    logger.info(f"  POST   http://{host}:{port}/resume              (CSRF protected)")
    logger.info(f"  POST   http://{host}:{port}/force_stack/ (CSRF protected)")
    logger.info(f"  GET    http://{host}:{port}/history")
    logger.info(f"  GET    http://{host}:{port}/health")
    logger.info("=" * 80 + "\n")

    # ``threaded=True`` allows Flask to serve multiple requests concurrently.
    app.run(host=host, port=port, debug=False, threaded=True)


def start_in_background() -> threading.Thread:
    """Launch the webhook server in a daemon thread."""
    thread = threading.Thread(target=_run_flask, daemon=True)
    thread.start()
    logger.info("✅ Webhook server started in background thread")
    return thread
def stop_server():
    """
    Shutdown helper – the surrounding process (Docker / systemd) will
    terminate the whole Python process.  This function exists mainly for
    completeness (e.g. unit‑testing or a graceful stop signal) and logs the
    intent before exiting.
    """
    logger.info("⚡️ Webhook server shutdown requested – exiting process")
    sys.exit(0)


# ----------------------------------------------------------------------
# Run directly (useful for local testing / debugging)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    #  Safety check – make sure the two secrets have been replaced
    # --------------------------------------------------------------
    if (
        WebhookConfig.WEBHOOK_TOKEN == "CHANGE_ME_TO_A_STRONG_RANDOM_TOKEN"
        or WebhookConfig.WEBHOOK_SECRET == "CHANGE_ME_TO_A_STRONG_RANDOM_HMAC_SECRET"
    ):
        logger.error(
            "\n"
            + "=" * 80
            + "\n"
            + "⚠️  SECURITY WARNING!\n"
            + "You must replace WEBHOOK_TOKEN and WEBHOOK_SECRET with strong, unique values.\n"
            + "Generate them with:\n"
            + '  python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            + "=" * 80
            + "\n"
        )
        sys.exit(1)

    # --------------------------------------------------------------
    #  Start the Flask server in the foreground (production mode)
    # --------------------------------------------------------------
    _run_flask()

