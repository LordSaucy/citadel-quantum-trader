# src/data_ingest/collector.py
import datetime as dt
import logging
import pandas as pd
import MetaTrader5 as mt5
from .config import load
from .utils import write_parquet, retry

log = logging.getLogger("cqt.ingest")

@retry(attempts=3, backoff=2.0)
def fetch_one_day(symbol: str, day: dt.date) -> pd.DataFrame:
    """
    Pull 1‑minute OHLCV for *symbol* on *day* from MT5.
    Returns a DataFrame with columns: time, open, high, low, close, tick_volume.
    """
    from_dt = dt.datetime.combine(day, dt.time.min)
    to_dt   = dt.datetime.combine(day, dt.time.max)

    # MT5 expects struct_time (seconds since epoch)
    rates = mt5.copy_rates_range(symbol,
                                 mt5.TIMEFRAME_M1,
                                 from_dt,
                                 to_dt)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No MT5 data for {symbol} on {day}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={
        "open":  "open",
        "high":  "high",
        "low":   "low",
        "close": "close",
        "tick_volume": "volume"
    })
    # Keep only the columns we need
    return df[["time", "open", "high", "low", "close", "volume"]]

def fetch_depth_snapshot(symbol: str) -> dict:
    """
    Optional – pull a shallow depth snapshot (best bid/ask, total volume).
    MT5 does not expose depth via the Python API, so you would need to
    call the broker’s REST endpoint (e.g., via the same credentials you
    use for order placement).  Stub left for future extension.
    """
    # Example placeholder:
    # resp = requests.get(f"https://api.broker.com/depth/{symbol}",
    #                     headers={"Authorization": f"Bearer {TOKEN}"})
    # return resp.json()
    return {}

def run():
    cfg = load()
    symbols = cfg["symbols"]
    provider = cfg.get("provider", "mt5")
    depth_needed = cfg.get("depth_snapshot", False)

    # Determine which day(s) to download.
    # For a daily cron you typically pull “yesterday”.
    today = dt.date.today()
    target_day = today - dt.timedelta(days=1)

    for sym in symbols:
        try:
            df = fetch_one_day(sym, target_day)
            write_parquet(df, sym, df["time"].iloc[0])
            if depth_needed:
                depth = fetch_depth_snapshot(sym)
                # Store depth as a tiny JSON alongside the parquet:
                depth_path = (DATA_ROOT /
                              f"{sym}_{target_day.strftime('%Y%m%d')}_depth.json")
                depth_path.write_text(json.dumps(depth))
        except Exception as exc:
            log.exception("Failed to ingest %s for %s", sym, target_day)
            # Depending on policy you may want to raise to abort the cron
            # or just continue to the next symbol.
