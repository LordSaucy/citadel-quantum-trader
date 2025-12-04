# src/data_ingest/collector.py
import datetime as dt
import logging
import pandas as pd
import MetaTrader5 as mt5
from .config import load
from .utils import write_parquet, retry
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

# -----------------------------------------------------------------
# Third‑party imports – install them in requirements.txt
# -----------------------------------------------------------------
import MetaTrader5 as mt5          # pip install MetaTrader5
import requests                    # pip install requests   (Polygon)

# -----------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------
from .utils import daterange, parquet_path
import yaml

log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Load configuration (YAML)
# -----------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing data‑ingest config: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

SYMBOLS: List[str] = CFG["symbols"]
LOOKBACK_DAYS: int = CFG.get("lookback_days", 30)
BASE_DATA_DIR = Path(__file__).parents[2] / "data"   # <repo_root>/data

# -----------------------------------------------------------------
# Helper – MT5 login (re‑usable for any MT5 call)
# -----------------------------------------------------------------
def _mt5_login() -> bool:
    cred = CFG.get("mt5", {})
    login = int(cred.get("login", 0))
    password = cred.get("password", "")
    server = cred.get("server", "")

    if not (login and password and server):
        log.error("MT5 credentials missing in config")
        return False

    if not mt5.initialize():
        log.error("MT5 init failed")
        return False

    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        log.error("MT5 login failed")
        mt5.shutdown()
    return authorized


# -----------------------------------------------------------------
# FETCHERS – each returns a DataFrame with columns:
#   time, open, high, low, close, tick_volume, spread, real_volume
# -----------------------------------------------------------------
def fetch_mt5(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Pull minute‑resolution OHLCV from MT5 using `copy_rates_range`.
    Returns a pandas DataFrame indexed by UTC timestamps.
    """
    if not _mt5_login():
        raise RuntimeError("Cannot login to MT5")

    # MT5 timeframe constant for 1‑minute bars
    tf = mt5.TIMEFRAME_M1

    # MT5 expects Unix timestamps (seconds)
    utc_start = int(start.timestamp())
    utc_end   = int(end.timestamp())

    rates = mt5.copy_rates_range(symbol, tf, utc_start, utc_end)
# Inside fetch_mt5, after copying rates:
depth = mt5.market_book_get(symbol)   # returns a list of depth levels
# Convert to a DataFrame, store alongside OHLCV or in a separate parquet:
depth_df = pd.DataFrame(depth)
depth_path = parquet_path(BASE_DATA_DIR, f"{symbol}_depth", day)
depth_df.to_parquet(depth_path, compression="snappy")

    
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No MT5 data for {symbol} {start}-{end}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "open",
        "high": "high",
        "low":  "low",
        "close":"close",
        "tick_volume": "volume",
        "spread": "spread",
        "real_volume": "real_volume"
    })
    # Keep only the columns we need for back‑testing
    return df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]


def fetch_polygon(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Polygon.io minute bars (free tier – 5‑minute granularity for most symbols,
    but minute data is available for equities/crypto).  Adjust the endpoint
    if you need a different asset class.
    """
    api_key = CFG.get("polygon", {}).get("api_key")
    if not api_key:
        raise RuntimeError("Polygon API key missing in config")

    # Polygon expects the ticker in the form “EURUSD” → “X:EURUSD” for FX
    # (you may need to adapt the format for your subscription)
    ticker = f"X:{symbol}"

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["results"]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })
    # Polygon does not give spread; set to 0
    df["spread"] = 0
    df["real_volume"] = df["volume"]
    return df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]


# -----------------------------------------------------------------
# Dispatcher – map source name → fetch function
# -----------------------------------------------------------------
SOURCE_HANDLERS = {
    "mt5": fetch_mt5,
    "polygon": fetch_polygon,
    # "dukascopy": fetch_dukascopy,   # you can add later
}


def download_one_day(symbol: str, day: datetime) -> pd.DataFrame:
    """
    Wrapper that calls the appropriate fetcher for the configured source.
    """
    source = CFG.get("source", "mt5")
    handler = SOURCE_HANDLERS.get(source)
    if not handler:
        raise ValueError(f"Unsupported source '{source}'. Add a handler in collector.py")

    # The API expects start at 00:00:00 UTC and end at 23:59:59 UTC of the same day
    start_dt = datetime(day.year, day.month, day.day, tzinfo=day.tzinfo)
    end_dt   = start_dt + timedelta(days=1) - timedelta(seconds=1)

    log.info(f"Fetching {symbol} {start_dt.date()} from {source}")
    df = handler(symbol, start_dt, end_dt)
    return df


def main():
    """
    Entry‑point used by cron or Docker CMD.
    Loops over every symbol and every missing day in the look‑back window,
    writes a Parquet file per (symbol, day).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=LOOKBACK_DAYS - 1)

    for symbol in SYMBOLS:
        for day in daterange(start_date, today):
            out_path = parquet_path(BASE_DATA_DIR, symbol, day)

            # Skip if we already have the file (idempotent)
            if out_path.is_file():
                log.debug(f"Already have {out_path}, skipping")
                continue

            try:
                df = download_one_day(symbol, day)
                # Parquet compression (snappy) keeps files tiny
                df.to_parquet(out_path, compression="snappy", index=False)
                log.info(f"Wrote {out_path} ({len(df)} rows)")
            except Exception as exc:
                log.error(f"Failed to download {symbol} {day.date()}: {exc}")

if __name__ == "__main__":
    main()
 

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
