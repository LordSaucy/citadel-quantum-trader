# src/data_ingest/utils.py
import pandas as pd, pathlib, time, functools, logging

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
