# src/data_ingest/utils.py
import pandas as pd, pathlib, time, functools, logging
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

def daterange(start: datetime, end: datetime):
    """Yield each date (midnight) between start and end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def parquet_path(base_dir: Path, symbol: str, dt: datetime) -> Path:
    """
    Return the full path where the parquet for a given symbol/date should live.
    Example: data/EURUSD/EURUSD_2024-09-30.parquet
    """
    subdir = base_dir / symbol
    subdir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol}_{dt.strftime('%Y-%m-%d')}.parquet"
    return subdir / filename


DATA_ROOT = pathlib.Path(__file__).parents[2] / "data"   # repo_root/data/

def retry(attempts: int = 3, backoff: float = 2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            for i in range(attempts):
                try:
                    return fn(*args, **kw)
                except Exception as e:
                    if i == attempts - 1:
                        raise
                    logging.warning("Retry %s/%s – %s", i + 1, attempts, e)
                    time.sleep(backoff * (i + 1))
        return wrapper
    return deco

def write_parquet(df: pd.DataFrame, symbol: str, dt: pd.Timestamp):
    """Store a single‑day dataframe as data/<symbol>_YYYYMMDD.parquet."""
    date_str = dt.strftime("%Y%m%d")
    out_dir = DATA_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}_{date_str}.parquet"
    df.to_parquet(out_file, compression="snappy")
    logging.info("Wrote %s rows → %s", len(df), out_file)
