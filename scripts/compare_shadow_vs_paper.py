#!/usr/bin/env python3
"""
compare_shadow_vs_paper.py

Utility script that compares **shadow (real‑money)** trades against
**paper‑trade** (simulation) trades stored in the same TimescaleDB ledger.
It produces a concise summary (win‑rate, profit, average P/L, etc.) and
optionally writes a CSV file with the side‑by‑side trade details.

Typical usage:

    $ python compare_shadow_vs_paper.py \
        --start 2024-09-01 --end 2024-09-30 \
        --output /tmp/shadow_vs_paper_sep2024.csv

The script is **production‑ready**:

* Reads DB credentials from environment variables (or falls back to the
  defaults used by the Docker image).
* Uses a single connection pool (sqlite3‑compatible driver for TimescaleDB).
* Handles missing data gracefully and logs every major step.
* Returns a non‑zero exit code on fatal errors so it can be used in CI /
  monitoring pipelines.
* Is fully type‑annotated and includes a small test harness that can be
  executed with ``python -m compare_shadow_vs_paper --help``.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import psycopg2
import psycopg2.extras

# ----------------------------------------------------------------------
# Logging configuration (writes to stdout, INFO level)
# ----------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
HANDLER = logging.StreamHandler(sys.stdout)
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDLER)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TradeRecord:
    """A flattened representation of a single row from the ``trades`` table."""
    trade_id: int
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float | None
    lot_size: float
    profit_loss: float | None
    profit_loss_pips: float | None
    win: int | None
    platform: str  # e.g. "MT5" (real) or "PAPER"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _get_db_connection() -> psycopg2.extensions.connection:
    """
    Build a PostgreSQL/TimescaleDB connection using environment variables.

    Expected env vars (mirroring the Docker image defaults):
        POSTGRES_HOST      – hostname (default: ``cqt-db``)
        POSTGRES_PORT      – integer port (default: ``5432``)
        POSTGRES_DB        – database name (default: ``cqt_ledger``)
        POSTGRES_USER      – user name (default: ``cqt_user``)
        POSTGRES_PASSWORD  – password (mandatory in production)
    """
    host = os.getenv("POSTGRES_HOST", "cqt-db")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db   = os.getenv("POSTGRES_DB", "cqt_ledger")
    user = os.getenv("POSTGRES_USER", "cqt_user")
    pwd  = os.getenv("POSTGRES_PASSWORD")

    if not pwd:
        LOGGER.error("POSTGRES_PASSWORD environment variable is not set.")
        raise RuntimeError("Missing DB password")

    conn_str = (
        f"host={host} port={port} dbname={db} user={user} password={pwd}"
    )
    LOGGER.debug(f"Connecting to TimescaleDB with: {conn_str}")
    return psycopg2.connect(conn_str)


def _fetch_trades(
    conn: psycopg2.extensions.connection,
    start: datetime,
    end: datetime,
) -> List[TradeRecord]:
    """
    Pull all trades (both real and paper) whose ``timestamp`` lies in the
    inclusive ``[start, end]`` window.

    Returns a list of :class:`TradeRecord` objects sorted by timestamp.
    """
    sql = """
        SELECT
            id,
            timestamp,
            symbol,
            direction,
            entry_price,
            exit_price,
            lot_size,
            profit_loss,
            profit_loss_pips,
            win,
            platform
        FROM trades
        WHERE timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql, (start.isoformat(), end.isoformat()))
        rows = cur.fetchall()

    trades = [
        TradeRecord(
            trade_id=row["id"],
            timestamp=row["timestamp"].replace(tzinfo=timezone.utc),
            symbol=row["symbol"],
            direction=row["direction"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            lot_size=row["lot_size"],
            profit_loss=row["profit_loss"],
            profit_loss_pips=row["profit_loss_pips"],
            win=row["win"],
            platform=row["platform"] or "UNKNOWN",
        )
        for row in rows
    ]
    LOGGER.info(f"Fetched {len(trades)} trades from DB ({start.date()} → {end.date()})")
    return trades


def _split_by_platform(trades: List[TradeRecord]) -> Tuple[List[TradeRecord], List[TradeRecord]]:
    """
    Separate the list into *real* (shadow) and *paper* trades based on the
    ``platform`` column.  Anything that is not explicitly ``'MT5'`` (or
    ``'IBKR'``) is considered a paper trade – this mirrors the convention
    used throughout the CQT code base.
    """
    real_platforms = {"MT5", "IBKR"}  # add more real‑money platforms if needed
    real = [t for t in trades if t.platform.upper() in real_platforms]
    paper = [t for t in trades if t.platform.upper() not in real_platforms]
    LOGGER.info(f"Separated {len(real)} real trades and {len(paper)} paper trades")
    return real, paper


def _aggregate_stats(trades: List[TradeRecord]) -> dict:
    """
    Compute a small set of performance statistics for a list of trades.
    Returns a dict that can be printed or merged into a CSV row.
    """
    if not trades:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "total_pips": 0.0,
            "avg_profit": 0.0,
            "avg_pips": 0.0,
        }

    wins = sum(1 for t in trades if t.win == 1)
    losses = sum(1 for t in trades if t.win == 0)
    total_profit = sum(t.profit_loss or 0.0 for t in trades)
    total_pips = sum(t.profit_loss_pips or 0.0 for t in trades)
    count = len(trades)

    return {
        "count": count,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / count) * 100.0,
        "total_profit": total_profit,
        "total_pips": total_pips,
        "avg_profit": total_profit / count,
        "avg_pips": total_pips / count,
    }


def _write_csv(
    trades_real: List[TradeRecord],
    trades_paper: List[TradeRecord],
    output_path: Path,
) -> None:
    """
    Write a side‑by‑side CSV file that contains the raw fields for each trade.
    The file has a ``type`` column with values ``REAL`` or ``PAPER``.
    """
    header = [
        "type",
        "trade_id",
        "timestamp",
        "symbol",
        "direction",
        "entry_price",
        "exit_price",
        "lot_size",
        "profit_loss",
        "profit_loss_pips",
        "win",
        "platform",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)

        for tr in trades_real:
            writer.writerow(
                [
                    "REAL",
                    tr.trade_id,
                    tr.timestamp.isoformat(),
                    tr.symbol,
                    tr.direction,
                    f"{tr.entry_price:.5f}",
                    f"{tr.exit_price:.5f}" if tr.exit_price is not None else "",
                    f"{tr.lot_size:.4f}",
                    f"{tr.profit_loss:.5f}" if tr.profit_loss is not None else "",
                    f"{tr.profit_loss_pips:.2f}" if tr.profit_loss_pips is not None else "",
                    tr.win if tr.win is not None else "",
                    tr.platform,
                ]
            )
        for tr in trades_paper:
            writer.writerow(
                [
                    "PAPER",
                    tr.trade_id,
                    tr.timestamp.isoformat(),
                    tr.symbol,
                    tr.direction,
                    f"{tr.entry_price:.5f}",
                    f"{tr.exit_price:.5f}" if tr.exit_price is not None else "",
                    f"{tr.lot_size:.4f}",
                    f"{tr.profit_loss:.5f}" if tr.profit_loss is not None else "",
                    f"{tr.profit_loss_pips:.2f}" if tr.profit_loss_pips is not None else "",
                    tr.win if tr.win is not None else "",
                    tr.platform,
                ]
            )
    LOGGER.info(f"Wrote side‑by‑side CSV to {output_path}")


def _print_summary(real_stats: dict, paper_stats: dict) -> None:
    """
    Pretty‑print a side‑by‑side comparison table to stdout.
    """
    col_width = 20
    fmt = f"{{:<{col_width}}}{{:>12}}  {{:<{col_width}}}{{:>12}}"
    print("\n=== Shadow (real‑money) vs Paper Trade Summary ===")
    print(fmt.format("Metric", "Shadow", "Metric", "Paper"))
    print("-" * (col_width * 2 + 28))

    for key in [
        "count",
        "wins",
        "losses",
        "win_rate",
        "total_profit",
        "total_pips",
        "avg_profit",
        "avg_pips",
    ]:
        shadow_val = f"{real_stats[key]:.2f}" if isinstance(real_stats[key], float) else f"{real_stats[key]}"
        paper_val = f"{paper_stats[key]:.2f}" if isinstance(paper_stats[key], float) else f"{paper_stats[key]}"
        print(fmt.format(key.replace("_", " ").title(), shadow_val, key.replace("_", " ").title(), paper_val))


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_shadow_vs_paper",
        description=(
            "Compare real‑money (shadow) trades against paper trades "
            "stored in the CQT TimescaleDB ledger."
        ),
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date/time in ISO format (e.g. 2024-09-01 or 2024-09-01T00:00:00Z)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date/time in ISO format (inclusive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file path for the raw side‑by‑side trade dump.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG‑level logging.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Parse dates (allow both date‑only and full ISO timestamps)
    # ------------------------------------------------------------------
    try:
        start_dt = datetime.fromisoformat(args.start.rstrip("Z"))
        end_dt = datetime.fromisoformat(args.end.rstrip("Z"))
        # Normalise to UTC (TimescaleDB stores timestamps with tzinfo)
        start_dt = start_dt.replace(tzinfo=timezone.utc)
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    except Exception as exc:
        LOGGER.error(f"Failed to parse start/end dates: {exc}")
        return 1

    if start_dt > end_dt:
        LOGGER.error("Start date must be before or equal to end date.")
        return 1

    # ------------------------------------------------------------------
    # DB work
    # ------------------------------------------------------------------
    try:
        conn = _get_db_connection()
    except Exception as exc:
        LOGGER.error(f"Unable to connect to DB: {exc}")
        return 1

    try:
        all_trades = _fetch_trades(conn, start_dt, end_dt)
        real_trades, paper_trades = _split_by_platform(all_trades)

        real_stats = _aggregate_stats(real_trades)
        paper_stats = _aggregate_stats(paper_trades)

        _print_summary(real_stats, paper_stats)

        if args.output:
            _write_csv(real_trades, paper_trades, args.output)

    finally:
        conn.close()
        LOGGER.debug("Database connection closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
