import time
import redis
import math
from datetime import datetime, timedelta
from config_loader import Config
from prometheus_client import Counter

cfg = Config().settings
r = redis.StrictRedis(host=cfg.get("redis_host", "redis"),
                      port=cfg.get("redis_port", 6379),
                      db=cfg.get("redis_db", 0),
                      decode_responses=True)

# -------------------------------------------------
# Prometheus counters for audit / alerting
# -------------------------------------------------
trade_blocked_total = Counter(
    "trade_blocked_total",
    "Number of trades blocked by shock‑detector",
    ["reason"]
)

# -------------------------------------------------
# Core guard – returns (blocked:bool, reason:str)
# -------------------------------------------------
def should_block_trade(symbol: str) -> tuple[bool, str]:
    """Run all shock‑detector checks. Return True + reason if the trade must be aborted."""
    # ---- 1️⃣ News‑sentiment guard ----
    sentiment = r.get("sentiment:latest")
    if sentiment:
        try:
            s = float(sentiment)
            low = cfg["risk_shocks"]["sentiment"]["low"]
            high = cfg["risk_shocks"]["sentiment"]["high"]
            if s < low or s > high:
                trade_blocked_total.labels(reason="news_sentiment").inc()
                return True, "news_sentiment"
        except ValueError:
            pass  # malformed value – ignore

    # ---- 2️⃣ Macro‑calendar guard ----
    # Assume you have a table `macro_events` in PostgreSQL with columns (symbol, ts, impact)
    # For simplicity we just check a Redis key set by the macro‑calendar job:
    macro_flag = r.get(f"macro:block:{symbol}")  # e.g., "1" during the 30‑min window
    if macro_flag == "1":
        trade_blocked_total.labels(reason="macro_event").inc()
        return True, "macro_event"

    # ---- 3️⃣ Volatility‑spike guard (ATR) ----
    # Expect an ATR value stored in Redis: `atr:{symbol}`
    try:
        atr = float(r.get(f"atr:{symbol}") or 0)
        ema_atr = float(r.get(f"atr_ema30:{symbol}") or 0)
        mult = cfg["risk_shocks"]["atr_spike_multiplier"]
        if ema_atr > 0 and atr > mult * ema_atr:
            trade_blocked_total.labels(reason="vol_spike").inc()
            return True, "vol_spike"
    except (ValueError, TypeError):
        pass

    # ---- 4️⃣ Sudden price‑jump guard ----
    # Store the latest price in Redis: `price:{symbol}`
    # And a rolling window of recent prices in a sorted set `price_hist:{symbol}`
    try:
        now = time.time()
        win_secs = cfg["risk_shocks"]["price_jump"]["window_minutes"] * 60
        hist_key = f"price_hist:{symbol}"
        # Get oldest price in the window
        old = r.zrangebyscore(hist_key, now - win_secs, now - win_secs, start=0, num=1, withscores=False)
        latest = r.get(f"price:{symbol}")
        if old and latest:
            old_price = float(old[0])
            latest_price = float(latest)
            pct_move = abs(latest_price - old_price) / old_price * 100
            if pct_move > cfg["risk_shocks"]["price_jump"]["threshold_pct"]:
                trade_blocked_total.labels(reason="price_jump").inc()
                return True, "price_jump"
    except Exception:
        pass  # any parsing issue → ignore

    # ---- 5️⃣ Edge‑decay (win‑rate floor) ----
    # Assume a Prometheus gauge `bucket_winrate` exists; we can query it via the API
    # For simplicity we just read a cached value from Redis (updated by the edge‑decay job)
    winrate = r.get("edge:winrate")
    if winrate and float(winrate) < cfg["risk_shocks"]["winrate_floor"]:
        trade_blocked_total.labels(reason="winrate_floor").inc()
        return True, "winrate_floor"

    # Nothing triggered → allow the trade
    return False, ""
