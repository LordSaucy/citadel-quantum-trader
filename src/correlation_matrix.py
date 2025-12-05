# src/correlation_matrix.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging

log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Configuration (you can also move these to config.yaml)
# -----------------------------------------------------------------
LOOKBACK_DAYS = 30                # rolling window length
SYMBOLS = [                       # basket of assets you care about
    "EURUSD", "GBPUSD", "USDJPY",
    "XAUUSD", "WTIUSD", "SPX500", "US30"
]
TIMEFRAME = "1m"                  # the granularity you store (1‑minute is fine)

# -----------------------------------------------------------------
def get_close_prices(engine: "sqlalchemy.Engine") -> pd.DataFrame:
    """
    Returns a DataFrame indexed by timestamp with one column per symbol.
    Only the last LOOKBACK_DAYS of data are fetched.
    """
    end_ts   = datetime.now()
    start_ts = end_ts - timedelta(days=LOOKBACK_DAYS)

    sql = text(f"""
        SELECT ts, symbol, close
        FROM market_ticks
        WHERE timeframe = :tf
          AND ts BETWEEN :start AND :end
          AND symbol = ANY(:symbols)
        ORDER BY ts ASC;
    """)

    df = pd.read_sql(sql, engine,
                     params={"tf": TIMEFRAME,
                             "start": start_ts,
                             "end": end_ts,
                             "symbols": SYMBOLS})
    # Pivot so each symbol becomes a column
    df = df.pivot(index="ts", columns="symbol", values="close")
    # Forward‑fill missing minutes (e.g., market‑closed periods)
    df = df.ffill().dropna()
    return df


def rolling_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix over the whole DataFrame.
    Returns a symmetric N×N DataFrame where N = len(SYMBOLS).
    """
    corr = df.corr(method="pearson")
    # Force the diagonal to exactly 1.0 (numerical noise can make it 0.9999)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def average_correlation(corr: pd.DataFrame) -> float:
    """
    Returns the *off‑diagonal* average correlation.
    """
    n = corr.shape[0]
    # Sum of all elements minus the diagonal, then divide by N*(N‑1)
    total = corr.values.sum() - n   # subtract the N ones on the diagonal
    return total / (n * (n - 1))


def compute_and_store(engine: "sqlalchemy.Engine"):
    """
    Main entry point – called by a daily cron job.
    Stores the matrix and the average correlation in a dedicated table.
    """
    df   = get_close_prices(engine)
    corr = rolling_corr_matrix(df)
    avg  = average_correlation(corr)

    # Persist the matrix as JSON (PostgreSQL jsonb column)
    matrix_json = corr.to_json(orient="split")   # includes index/columns/values

    with engine.begin() as conn:
        # Create the table if it does not exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS correlation_snapshots (
                ts          TIMESTAMP PRIMARY KEY,
                avg_corr    DOUBLE PRECISION,
                matrix_json JSONB
            );
        """))

        conn.execute(
            text("""
                INSERT INTO correlation_snapshots (ts, avg_corr, matrix_json)
                VALUES (:ts, :avg, :matrix)
                ON CONFLICT (ts) DO UPDATE
                SET avg_corr = EXCLUDED.avg_corr,
                    matrix_json = EXCLUDED.matrix_json;
            """),
            {"ts": datetime.now(),
             "avg": avg,
             "matrix": matrix_json}
        )
    log.info(f"[Corr] Avg correlation = {avg:.3f}")
    return avg, corr
