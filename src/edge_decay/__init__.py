# src/edge_decay/__init__.py
import logging

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Try to import the real implementations – fall back to a no‑op stub.
# ------------------------------------------------------------------
try:
    from src.watchdog import activate_edge_decay as _wd_activate
except Exception:   # pragma: no cover
    _wd_activate = None

try:
    from src.edge_decay import trigger_edge_decay as _ed_trigger
except Exception:   # pragma: no cover
    _ed_trigger = None

def trigger_edge_decay(reason: str) -> None:
    """
    Unified entry point used by the drift‑detector.

    It will call every known implementation (watchdog, dedicated module,
    etc.).  If none are available it simply logs a warning.
    """
    if _wd_activate:
        try:
            _wd_activate(reason)
            log.info("[edge‑decay] watchdog notified")
        except Exception as exc:   # pragma: no cover
            log.error("[edge‑decay] watchdog failed: %s", exc)

    if _ed_trigger:
        try:
            _ed_trigger(reason)
            log.info("[edge‑decay] dedicated detector notified")
        except Exception as exc:   # pragma: no cover
            log.error("[edge‑decay] detector failed: %s", exc)

    if not (_wd_activate or _ed_trigger):
        log.warning("[edge‑decay] no real implementation found – only logging")
