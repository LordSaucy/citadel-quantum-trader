# src/utils/common.py
from datetime import datetime, timezone

def utc_now() -> datetime:
    """Return a timezone‑aware UTC datetime (single source of truth)."""
    return datetime.now(timezone.utc)
