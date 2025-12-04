#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/main.py

Unified entry‑point for the Citadel Quantum Trader (CQT) engine.

* FastAPI  – control API, health checks, Prometheus metrics, static UI.
* Flask    – legacy ConfluenceController (exposes /config/*).
* Background services:
    – DOM cache (single subscription to MT5/IBKR depth feed)
    – Config hot‑reloader (reloads config.yaml on change)
    – Prometheus exporter (exposes custom gauges)
* Graceful shutdown handling.
"""

# -------------------------------------------------------------------------
# 0️⃣  Standard library & third‑party imports
# -------------------------------------------------------------------------
import os
import sys
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.wsgi import WSGIMiddleware
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    start_http_server,
)

# -------------------------------------------------------------------------
# 1️⃣  Internal imports (keep them grouped logically)
# -------------------------------------------------------------------------
# FastAPI routers ---------------------------------------------------------
from .api.health   import router as health_router
from .api.config   import router as config_router
from .api.risk     import router as risk_router
from .api.bot      import router as bot_router
from .api.orders   import router as orders_router
from .api.trades   import router as trades_router
from .api.backup   import router as backup_router
from .api.mode     import router as mode_router
from .api.session  import router as session_router
from .api.admin    import router as admin_router

# Flask (legacy ConfluenceController) ------------------------------------
from .backend.app.main import create_app as create_flask_app   # <-- Flask factory
from .backend.app.auth import router as auth_router          # optional FastAPI auth

# Core engine components --------------------------------------------------
from .market_data.dom_cache import DomCache
from .market_data.lir       import compute_lir
from .risk_management.risk_manager import RiskManager
from .execution_engine.execution_engine import ExecutionEngine
from .signal_engine.signal_processor import SignalProcessor
from .utils.config_loader   import load_config
from .utils.shutdown        import register_graceful_shutdown

# -------------------------------------------------------------------------
# 2️⃣  Global objects (singletons) – created once at import time
# -------------------------------------------------------------------------
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# FastAPI app -------------------------------------------------------------
app = FastAPI(
    title="Citadel Quantum Trader (CQT) Control API",
    version=os.getenv("CQT_VERSION", "dev"),
    description=(
        "Internal‑only control plane for the Citadel Quantum Trader. "
        "All endpoints require a bearer token (`Authorization: Bearer <token>`)."
    ),
)

# -------------------------------------------------------------------------
# 3️⃣  CORS – allow Grafana (or local dev) to call the API from the browser
# -------------------------------------------------------------------------
origins = [
    "https://grafana.cqt.example.com",
    "http://localhost:3000",   # local dev
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# 4️⃣  Register FastAPI routers (order does not matter)
# -------------------------------------------------------------------------
app.include_router(health_router)
app.include_router(config_router)
app.include_router(risk_router)
app.include_router(bot_router)
app.include_router(orders_router)
app.include_router(trades_router)
app.include_router(backup_router)
app.include_router(mode_router)
app.include_router(session_router)
app.include_router(admin_router)

# -------------------------------------------------------------------------
# 5️⃣  Mount the legacy Flask app under `/legacy` (so `/config/*` works)
# -------------------------------------------------------------------------
flask_app = create_flask_app()
app.mount("/legacy", WSGIMiddleware(flask_app))

# -------------------------------------------------------------------------
# 6️⃣  Prometheus custom metrics (LIR, total depth, etc.)
# -------------------------------------------------------------------------
lir_gauge   = Gauge(
    "cqt_liquidity_imbalance_ratio",
    "Liquidity‑Imbalance Ratio per bucket/symbol",
    ["bucket_id", "symbol"],
)
depth_gauge = Gauge(
    "cqt_total_market_depth",
    "Bid + Ask volume for the top N levels (units)",
    ["bucket_id", "symbol"],
)

# -------------------------------------------------------------------------
# 7️⃣  Background services
# -------------------------------------------------------------------------

# 7.1  DOM cache – subscribe once at startup, keep latest depth in memory
def _start_dom_cache():
    """Initialise the singleton DOM cache with the broker connection."""
    # Choose the broker implementation you need (IBKR or MT5)
    # The connector reads credentials from Vault / env vars.
    from ibkr_dom import connect_ibkr   # or: from mt5_dom import connect_mt5
    broker = connect_ibkr()            # <-- reads IBKR_HOST, USER, PASS
    DomCache(broker_interface=broker, top_n=20)
    log.info("DOM cache started (top 20 levels per side)")

# 7.2  Config hot‑reloader – watches config.yaml and reloads the in‑memory config
def _start_config_watcher():
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    last_mtime = cfg_path.stat().st_mtime

    def _watcher():
        nonlocal last_mtime
        while True:
            time.sleep(5)
            try:
                cur_mtime = cfg_path.stat().st_mtime
                if cur_mtime != last_mtime:
                    load_config()               # re‑load the YAML into the singleton
                    last_mtime = cur_mtime
                    log.info("[CONFIG] Reloaded from config.yaml")
            except Exception as exc:
                log.error("[CONFIG] Watcher error: %s", exc)

    threading.Thread(target=_watcher, daemon=True, name="cfg-watcher").start()

# 7.3  Prometheus HTTP endpoint (exposed on 0.0.0.0:9090)
def _start_prometheus():
    start_http_server(9090)   # already mapped in docker‑compose.yml
    log.info("Prometheus metrics endpoint started on :9090")

# -------------------------------------------------------------------------
# 8️⃣  Application startup & shutdown events
# -------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    log.info("🚀 CQT FastAPI control plane starting …")
    _start_prometheus()
    _start_dom_cache()
    _start_config_watcher()
    register_graceful_shutdown()   # registers SIGTERM/SIGINT handlers
    log.info("✅ Startup complete – API ready at /docs")

@app.on_event("shutdown")
async def on_shutdown():
    log.info("🛑 CQT shutting down – cleaning up resources …")
    # If you have any explicit close() methods on singletons, call them here.
    # Example:
    # DomCache().close()
    log.info("✅ Shutdown complete.")

# -------------------------------------------------------------------------
# 9️⃣  Root endpoint – friendly welcome message
# -------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "CQT Control API – visit /docs for the OpenAPI UI"}

# -------------------------------------------------------------------------
# 10️⃣  Helper: expose Prometheus metrics on the FastAPI side as well
# -------------------------------------------------------------------------
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrapes this endpoint (in addition to the raw :9090 port)."""
    data = generate_latest()
    return StreamingResponse(
        iter([data]),
        media_type=CONTENT_TYPE_LATEST,
    )

# -------------------------------------------------------------------------
# 11️⃣  OPTIONAL: Simple health endpoint for the load‑balancer
# -------------------------------------------------------------------------
@app.get("/healthz", response_model=dict)
async def healthz():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

# -------------------------------------------------------------------------
# 12️⃣  If you run the module directly (local dev), start Uvicorn
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Respect optional HOST/PORT env vars (useful for local testing)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8005"))
    uvicorn.run("src.main:app", host=host, port=port, log_level="info")
