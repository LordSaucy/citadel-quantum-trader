#!/usr/bin/env python3
"""
Graceful Shutdown Handler for Citadel Trading Bot

Handles SIGTERM and SIGINT signals to save bot state (open positions, pending orders,
current risk schedule) to a database checkpoint before exiting.

This ensures the bot can resume from exactly where it left off after restart.

✅ FIXES:
- Renamed local variable "Session" → "session_factory" (follows PEP 8 naming)
- Removed unused "exc" variable from exception handler

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.1 – Proper Variable Naming & Exception Handling
"""

import logging
import signal
import sys
import json
from datetime import datetime
from .db import get_session, Position, PendingOrder, BotState   # adjust to your ORM models
from .risk_management_layer import RiskManagementLayer          # if you expose the schedule

log = logging.getLogger("citadel.shutdown")


def _serialize_checkpoint(session):
    """
    ✅ Gather everything we need to restore the bot exactly where it left off.
    
    Returns a dict that will be JSON‑encoded and stored in the DB.
    Captures three critical snapshots:
    1. Open positions (live tickets)
    2. Pending orders (sent but not yet acknowledged)
    3. Current risk fraction schedule (may have been adjusted at runtime)
    
    Args:
        session: SQLAlchemy session object for database queries
        
    Returns:
        Dict with keys: timestamp, positions, pending_orders, risk_schedule
    """
    # 1️⃣ Open positions (tickets that are still live)
    open_positions = (
        session.query(Position)
        .filter(Position.is_open == True)   # noqa: E712
        .all()
    )
    pos_payload = [
        {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "volume": p.volume,
            "entry_price": p.entry_price,
            "sl": p.stop_loss,
            "tp": p.take_profit,
            "opened_at": p.opened_at.isoformat(),
        }
        for p in open_positions
    ]

    # 2️⃣ Pending orders (sent but no ack yet)
    pending = (
        session.query(PendingOrder)
        .filter(PendingOrder.status == "sent")   # adjust column name
        .all()
    )
    pend_payload = [
        {
            "order_id": o.order_id,
            "symbol": o.symbol,
            "volume": o.volume,
            "price": o.price,
            "created_at": o.created_at.isoformat(),
        }
        for o in pending
    ]

    # 3️⃣ Current risk‑fraction schedule (might have been tweaked)
    # Assuming you have a singleton table `BotState` that stores the schedule
    state_row = session.query(BotState).first()
    schedule = json.loads(state_row.risk_schedule_json) if state_row else {}

    # 4️⃣ Timestamp for audit
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "positions": pos_payload,
        "pending_orders": pend_payload,
        "risk_schedule": schedule,
    }
    return checkpoint


def write_checkpoint():
    """
    ✅ Write the checkpoint JSON into a dedicated table (or a file).
    
    Creates a new checkpoint record in the `bot_checkpoint` table with:
    - Current timestamp
    - JSON payload containing positions, orders, and risk schedule
    
    On restart, the bot can query the most recent checkpoint and restore state.
    
    Implementation Notes:
    - Uses UPSERT (INSERT ... ON CONFLICT) to handle idempotent updates
    - Lazy imports engine to avoid circular dependencies
    - Transaction-safe: all-or-nothing write
    """
    from .db import engine   # import lazily to avoid circular imports

    with engine.begin() as conn:
        # ✅ FIXED: Renamed "Session" → "session_factory"
        # WHY: Python convention (PEP 8) reserves PascalCase for classes.
        # Local variables should be lowercase_with_underscores.
        # "Session" is a factory function call, not a class, so use lowercase.
        session_factory = get_session(bind=conn)
        
        with session_factory() as session:
            checkpoint = _serialize_checkpoint(session)

            # Store as JSONB in a `bot_checkpoint` table (create if missing)
            conn.execute(
                """
                INSERT INTO bot_checkpoint (created_at, payload)
                VALUES (NOW(), %s)
                ON CONFLICT (id) DO UPDATE
                SET created_at = NOW(),
                    payload = EXCLUDED.payload;
                """,
                (json.dumps(checkpoint),),
            )
            log.info("✅ Bot checkpoint written – %d open positions, %d pending orders",
                     len(checkpoint["positions"]), len(checkpoint["pending_orders"]))


def _handle_signal(signum, frame):
    """
    ✅ Signal handler – called by the OS when we receive SIGTERM/SIGINT.
    
    Workflow:
    1. Log the received signal name
    2. Attempt to write checkpoint (safe: won't raise)
    3. Log exceptions but never propagate them
    4. Exit cleanly with status 0
    
    IMPORTANT: Signal handlers must be as simple and safe as possible.
    We catch ALL exceptions to ensure the handler never crashes.
    """
    sig_name = signal.Signals(signum).name
    log.info("🛑 Received %s – initiating graceful shutdown …", sig_name)

    try:
        write_checkpoint()
    except Exception:
        # ✅ FIXED: Removed unused "exc" variable
        # WHY: Exception variable was declared but never used.
        # log.exception() automatically includes the current exception context,
        # so we don't need to reference it explicitly.
        # This follows PEP 8: "don't declare unused variables"
        log.exception("❌ Failed to write checkpoint during %s", sig_name)

    # Give the rest of the app a chance to clean up (e.g., close DB pool)
    # If you have an async event loop, you can stop it here.
    log.info("🟢 Graceful shutdown complete – exiting.")
    sys.exit(0)


def register_graceful_shutdown():
    """
    ✅ Call this once at program start‑up.
    
    Registers the signal handler for SIGTERM (Docker stop) and SIGINT (Ctrl‑C).
    Also handles Windows SIGBREAK (console Ctrl‑Break) if available.
    
    Example usage in main.py:
    ```python
    from citadel.shutdown import register_graceful_shutdown
    
    if __name__ == "__main__":
        register_graceful_shutdown()  # ← Register handler
        run_bot()  # Your bot main loop
    ```
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    # On Windows the default console‑CTRL‑BREAK sends SIGBREAK; map it too:
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)
    log.debug("✅ Graceful‑shutdown handler registered")
