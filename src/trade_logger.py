#!/usr/bin/env python3
"""
TRADE LOGGER

Comprehensive trade logging system with database storage and analytics.
Tracks every trade with full details for performance analysis.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from .utils import dict_hash
from .config_loader import Config
import boto3
from prometheus_client import Gauge


# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/trade_logger.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "trade_logger.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# TradeLogger implementation
# ----------------------------------------------------------------------
class TradeLogger:
    """
    Advanced trade logging system with SQLite database.
    Stores all trade data for analysis and reporting.
    """

    # ------------------------------------------------------------------
    # Construction / DB initialisation
    # ------------------------------------------------------------------
    def __init__(self, db_path: str = "trades.db"):
        """
        Initialise trade logger.

        Args:
            db_path: Path to SQLite database file (relative to the cwd)
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self._init_database()
        logger.info(f"Trade Logger initialised with DB: {self.db_path}")

    def _init_database(self) -> None:
        """Create the required tables if they do not already exist."""
        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        cursor = self.connection.cursor()

        # ----------------------------------------------------------------
        # Trades table – one row per order (open → close)
        # ----------------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                symbol            TEXT    NOT NULL,
                direction         TEXT    NOT NULL,
                entry_price       REAL    NOT NULL,
                exit_price        REAL,
                lot_size          REAL    NOT NULL,
                stop_loss         REAL    NOT NULL,
                take_profit       REAL    NOT NULL,
                entry_quality     INTEGER,
                confluence_score  INTEGER,
                mtf_alignment     REAL,
                market_regime     TEXT,
                session           TEXT,
                volatility_state  TEXT,
                rr_ratio          REAL,
                stack_level       INTEGER,
                profit_loss       REAL,
                profit_loss_pips  REAL,
                win               INTEGER,
                exit_reason       TEXT,
                platform          TEXT,
                magic_number      INTEGER,
                comment           TEXT,
                metadata          TEXT
            )
            """
        )

        # ----------------------------------------------------------------
        # Daily performance aggregates
        # ----------------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_performance (
                date            TEXT PRIMARY KEY,
                total_trades    INTEGER,
                wins            INTEGER,
                losses          INTEGER,
                win_rate        REAL,
                total_profit    REAL,
                total_pips      REAL,
                best_trade      REAL,
                worst_trade     REAL,
                average_win     REAL,
                average_loss    REAL,
                profit_factor   REAL,
                sharpe_ratio    REAL
            )
            """
        )

        # ----------------------------------------------------------------
        # System‑wide metrics (heartbeat / health)
        # ----------------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp          TEXT PRIMARY KEY,
                levers_active      INTEGER,
                platform_status    TEXT,
                risk_utilization   REAL,
                drawdown_pct       REAL,
                equity             REAL,
                balance            REAL,
                open_positions     INTEGER,
                metadata           TEXT
            )
            """
        )

        self.connection.commit()
        logger.info("Database tables created / verified")

    # ------------------------------------------------------------------
    # Trade open / close logging
    # ------------------------------------------------------------------
    def log_trade_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        entry_quality: int = 0,
        confluence_score: int = 0,
        mtf_alignment: float = 0.0,
        market_regime: str = "UNKNOWN",
        session: str = "UNKNOWN",
        volatility_state: str = "NORMAL",
        rr_ratio: float = 2.0,
        stack_level: int = 1,
        platform: str = "MT5",
        magic_number: int = 0,
        comment: str = "",
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Record a newly opened trade.

        Returns the autogenerated ``trade_id``.
        """
        cursor = self.connection.cursor()
        timestamp = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        cursor.execute(
            """
            INSERT INTO trades (
                timestamp, symbol, direction, entry_price, lot_size,
                stop_loss, take_profit, entry_quality, confluence_score,
                mtf_alignment, market_regime, session, volatility_state,
                rr_ratio, stack_level, platform, magic_number,
                comment, metadata
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                timestamp,
                symbol,
                direction,
                entry_price,
                lot_size,
                stop_loss,
                take_profit,
                entry_quality,
                confluence_score,
                mtf_alignment,
                market_regime,
                session,
                volatility_state,
                rr_ratio,
                stack_level,
                platform,
                magic_number,
                comment,
                metadata_json,
            ),
        )
        self.connection.commit()
        trade_id = cursor.lastrowid
        logger.info(
            f"Trade opened #{trade_id}: {direction} {symbol} @ {entry_price:.5f} "
            f"({lot_size:.2f} lots, EQ={entry_quality}, CONF={confluence_score})"
        )
        return trade_id

    def log_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        profit_loss: float,
        profit_loss_pips: float,
        exit_reason: str = "TP",
    ) -> None:
        """
        Record the closing of a trade.

        ``profit_loss`` is expressed in the account currency,
        ``profit_loss_pips`` in pips.
        """
        cursor = self.connection.cursor()
        win_flag = 1 if profit_loss > 0 else 0

        cursor.execute(
            """
            UPDATE trades
            SET exit_price = ?, profit_loss = ?, profit_loss_pips = ?,
                win = ?, exit_reason = ?
            WHERE id = ?
            """,
            (exit_price, profit_loss, profit_loss_pips, win_flag, exit_reason, trade_id),
        )
        self.connection.commit()

        result = "WIN" if win_flag else "LOSS"
        logger.info(
            f"Trade closed #{trade_id}: {result} ${profit_loss:.2f} "
            f"({profit_loss_pips:.1f} pips) – {exit_reason}"
        )

        # Re‑calculate the daily performance aggregates
        self._update_daily_performance()

    # ------------------------------------------------------------------
    # Daily performance aggregation
    # ------------------------------------------------------------------
    def _update_daily_performance(self) -> None:
        """Re‑compute the aggregated stats for the current calendar day."""
        cursor = self.connection.cursor()
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Grab all closed trades for today
        cursor.execute(
            """
            SELECT profit_loss, profit_loss_pips, win
            FROM trades
            WHERE DATE(timestamp) = ?
              AND exit_price IS NOT NULL
            """,
            (today,),
        )
        rows = cursor.fetchall()
        if not rows:
            return

        total_trades = len(rows)
        wins = sum(1 for r in rows if r[2] == 1)
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100.0
        total_profit = sum(r[0] for r in rows)
        total_pips = sum(r[1] for r in rows)

        winning_pl = [r[0] for r in rows if r[2] == 1]
        losing_pl = [r[0] for r in rows if r[2] == 0]

        best_trade = max(winning_pl) if winning_pl else 0.0
        worst_trade = min(losing_pl) if losing_pl else 0.0
        avg_win = sum(winning_pl) / len(winning_pl) if winning_pl else 0.0
        avg_loss = sum(losing_pl) / len(losing_pl) if losing_pl else 0.0

        gross_profit = sum(winning_pl) if winning_pl else 0.0
        gross_loss = abs(sum(losing_pl)) if losing_pl else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        cursor.execute(
            """
            INSERT OR REPLACE INTO daily_performance (
                date, total_trades, wins, losses, win_rate,
                total_profit, total_pips, best_trade, worst_trade,
                average_win, average_loss, profit_factor
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                today,
                total_trades,
                wins,
                losses,
                win_rate,
                total_profit,
                total_pips,
                best_trade,
                worst_trade,
                avg_win,
                avg_loss,
                profit_factor,
            ),
        )
        self.connection.commit()
        logger.debug(f"Daily performance for {today} refreshed")

    # ------------------------------------------------------------------
    # System‑wide metrics (heartbeat / health)
    # ------------------------------------------------------------------
    def log_system_metrics(
        self,
        levers_active: int,
        platform_status: Dict,
        risk_utilization: float,
        drawdown_pct: float,
        equity: float,
        balance: float,
        open_positions: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record a snapshot of the engine / platform health.
        """
        cursor = self.connection.cursor()
        timestamp = datetime.utcnow().isoformat()
        platform_json = json.dumps(platform_status)
        meta_json = json.dumps(metadata) if metadata else "{}"

        cursor.execute(
            """
            INSERT INTO system_metrics (
                timestamp, levers_active, platform_status,
                risk_utilization, drawdown_pct, equity, balance,
                open_positions, metadata
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                timestamp,
                levers_active,
                platform_json,
                risk_utilization,
                drawdown_pct,
                equity,
                balance,
                open_positions,
                meta_json,
            ),
        )
        self.connection.commit()
        logger.debug("System metrics heartbeat recorded")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Return a high‑level performance snapshot for the last ``days`` days.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            f"""
            SELECT
                COUNT(*)                                     AS total_trades,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END)    AS wins,
                SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END)    AS losses,
                AVG(CASE WHEN win = 1 THEN 100.0 ELSE 0 END) AS win_rate,
                SUM(profit_loss)                            AS total_profit,
                SUM(profit_loss_pips)                       AS total_pips,
                AVG(entry_quality)                          AS avg_entry_quality,
                AVG(confluence_score)                       AS avg_confluence,
                AVG(mtf_alignment)                          AS avg_mtf_alignment
            FROM trades
            WHERE DATE(timestamp) >= DATE('now', ?)
              AND exit_price IS NOT NULL
            """,
            (f"-{days} days",),
        )
        row = cursor.fetchone()
        if row and row[0]:
            total = row[0]
            wins = row[1] or 0
            return {
                "total_trades": total,
                "wins": wins,
                "losses": row[2] or 0,
                "win_rate": (wins / total) * 100.0,
                "total_profit": row[4] or 0.0,
                "total_pips": row[5] or 0.0,
                "avg_entry_quality": row[6] or 0.0,
                "avg_confluence": row[7] or 0.0,
                "avg_mtf_alignment": row[8] or 0.0,
            }

        # No data – return empty defaults
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "total_pips": 0.0,
            "avg_entry_quality": 0.0,
            "avg_confluence": 0.0,
            "avg_mtf_alignment": 0.0,
        }

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """
        Fetch the most recent ``limit`` trades (open or closed).
        Returns a list of dictionaries – each key matches a column name.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT *
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Clean‑up helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the SQLite connection gracefully."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("TradeLogger DB connection closed")

    def __del__(self):
        """Ensure the DB is closed when the object is garbage‑collected."""
        self.close()

class TradeLogger:
    # ... existing methods ...

    def log_depth_check(self, bucket_id: int, symbol: str,
                        depth_ok: bool, agg_bid: float, agg_ask: float):
        entry = {
            "event": "depth_check",
            "bucket_id": bucket_id,
            "symbol": symbol,
            "depth_ok": depth_ok,
            "agg_bid_volume": agg_bid,
            "agg_ask_volume": agg_ask,
            "ts": datetime.utcnow().isoformat(),
        }
        self._store_entry(entry)   # same Merkle‑hash pipeline as other events

