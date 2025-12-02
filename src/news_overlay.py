import asyncio
import json
import os
import time
from collections import deque
import websockets

# -----------------------------------------------------------------
# Configuration (can be moved to config.yaml)
# -----------------------------------------------------------------
FINNHUB_WS_URL = "wss://ws.finnhub.io?token=" + os.getenv("FINNHUB_TOKEN")
IMPACT_THRESHOLD = 0.8          # only consider impact >= 0.8 as “high”
CANCEL_WINDOW_SEC = 30          # cancel signals that occur within 30 s after news

# -----------------------------------------------------------------
# In‑memory store of recent high‑impact news timestamps (deque for O(1) pops)
# -----------------------------------------------------------------
_recent_news = deque()   # each entry: (epoch_timestamp, impact)

async def _listener():
    async with websockets.connect(FINNHUB_WS_URL) as ws:
        # Subscribe to all symbols you trade – you can also subscribe to a wildcard
        # Finnhub does not support wildcards, so you must send a subscribe per symbol.
        # For simplicity we subscribe to the generic “news” channel (Finnhub sends all news)
        await ws.send(json.dumps({"type":"subscribe","symbol":"NEWS"}))

        async for message in ws:
            data = json.loads(message)
            for item in data.get("data", []):
                # Finnhub returns a dict with fields: headline, datetime, source, etc.
                impact = float(item.get("impact", 0.0))   # 0‑1 scale
                if impact >= IMPACT_THRESHOLD:
                    ts = time.time()
                    _recent_news.append((ts, impact))

# -----------------------------------------------------------------
# Background task starter – call once at app startup
# -----------------------------------------------------------------
def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.create_task(_listener())

# -----------------------------------------------------------------
# Public API – call from the signal pipeline
# -----------------------------------------------------------------
def is_recent_high_impact(signal_ts: float) -> bool:
    """
    signal_ts – Unix epoch (seconds) when the signal was generated.
    Returns True if a high‑impact news item arrived within CANCEL_WINDOW_SEC
    BEFORE the signal (i.e., news came first, then the signal).
    """
    # Purge old entries
    cutoff = time.time() - CANCEL_WINDOW_SEC
    while _recent_news and _recent_news[0][0] < cutoff:
        _recent_news.popleft()

    # If any news timestamp is *earlier* than the signal but within the window,
    # we consider the signal contaminated.
    for news_ts, _ in _recent_news:
        if news_ts <= signal_ts and (signal_ts - news_ts) <= CANCEL_WINDOW_SEC:
            return True
    return False
