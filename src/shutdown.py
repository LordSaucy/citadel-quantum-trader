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
    Gather everything we need to restore the bot exactly where it left off.
    Returns a dict that will be JSON‑encoded and stored in the DB.
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
    """Write the checkpoint JSON into a dedicated table (or a file)."""
    from .db import engine   # import lazily to avoid circular imports

    with engine.begin() as conn:
        Session = get_session(bind=conn)
        with Session() as session:
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
    """Signal handler – called by the OS when we receive SIGTERM/SIGINT."""
    sig_name = signal.Signals(signum).name
    log.info("🛑 Received %s – initiating graceful shutdown …", sig_name)

    try:
        write_checkpoint()
    except Exception as exc:   # never let an exception escape the handler
        log.exception("❌ Failed to write checkpoint during %s", sig_name)

    # Give the rest of the app a chance to clean up (e.g., close DB pool)
    # If you have an async event loop, you can stop it here.
    log.info("🟢 Graceful shutdown complete – exiting.")
    sys.exit(0)


def register_graceful_shutdown():
    """
    Call this once at program start‑up.
    It registers the handler for SIGTERM (Docker stop) and SIGINT (Ctrl‑C).
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    # On Windows the default console‑CTRL‑BREAK sends SIGBREAK; map it too:
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)
    log.debug("✅ Graceful‑shutdown handler registered")
