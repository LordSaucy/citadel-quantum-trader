# src/risk/session_manager.py
SESSION_RULES = {
    "forex": ("LONDON_NY_OVERLAP", (8, 16)),   # UTC
    "metal": ("NY_SESSION", (13, 21)),
    "index": ("US_SESSION", (13, 21)),
    "stock": ("US_SESSION", (13, 21)),
    "future": ("US_SESSION", (13, 21)),
    "crypto": ("ALL_DAY", (0, 24)),
}


def allowed_now(symbol: str, now_utc: datetime) -> bool:
    asset = classify_asset(symbol)
    _, (start, end) = SESSION_RULES[asset]
    hour = now_utc.hour
    return start <= hour < end
