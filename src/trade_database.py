#!/usr/bin/env python3
"""
Trade Database

SQLite database for trade tracking, analytics, and reporting.

Features:
    • Complete trade history
    • Performance analytics
    • Export functionality (JSON / CSV)
    • Rich query interface
    • System‑wide audit trail
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import csv
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Logging configuration (the application already configures a root logger;
# we just obtain a child logger here)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class TradeRecord:
    """Complete trade record – mirrors the `trades` table."""
    ticket: int
    symbol: str
    direction: str                     # "BUY" or "SELL"
    entry_time: str                    # ISO‑8601 timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_amount: float
    stack_level: int
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    r_multiple: Optional[float] = None
    quality_score: Optional[float] = None
    confluence_score: Optional[int] = None
    was_breakeven_moved: bool = False
    was_partial_closed: bool = False
    exit_reason: Optional[str] = None
    notes: Optional[str] = None


# ----------------------------------------------------------------------
# Core database wrapper
# ----------------------------------------------------------------------
class TradeDatabase:
    """
    SQLite‑backed trade database.

    All methods are deliberately tiny and raise **no** exceptions – they
    log the error and return ``False``/``None`` so the calling code can
    continue safely.
    """

    # ------------------------------------------------------------------
    # Construction / schema creation
    # ------------------------------------------------------------------
    def __init__(self, db_path: str = "trading_history.db"):
        """
        Initialise the database.

        Args:
            db_path: Path (relative or absolute) to the SQLite file.
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_database()
        logger.info(f"📊 Trade database initialised → {self.db_path}")

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Create (or reuse) a thread‑safe SQLite connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, isolation_level=None
            )
            self.conn.row_factory = sqlite3.Row
        return self.conn

    # ------------------------------------------------------------------
    def _initialize_database(self) -> None:
        """Create tables if they do not already exist."""
        conn = self._connect()
        cur = conn.cursor()

        # ----------------------------------------------------------------
        # Trades table – the core of the system
        # ----------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                ticket               INTEGER PRIMARY KEY,
                symbol               TEXT    NOT NULL,
                direction            TEXT    NOT NULL,
                entry_time           TEXT    NOT NULL,
                entry_price          REAL    NOT NULL,
                stop_loss            REAL    NOT NULL,
                take_profit          REAL    NOT NULL,
                lot_size             REAL    NOT NULL,
                risk_amount          REAL    NOT NULL,
                stack_level          INTEGER NOT NULL,
                exit_time            TEXT,
                exit_price           REAL,
                profit               REAL,
                r_multiple           REAL,
                quality_score        REAL,
                confluence_score     INTEGER,
                was_breakeven_moved INTEGER DEFAULT 0,
                was_partial_closed  INTEGER DEFAULT 0,
                exit_reason          TEXT,
                notes                TEXT,
                created_at           TEXT    DEFAULT (datetime('now'))
            );
            """
        )

        # ----------------------------------------------------------------
        # Account snapshots – useful for equity / draw‑down analysis
        # ----------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                balance       REAL    NOT NULL,
                equity        REAL    NOT NULL,
                margin_free   REAL    NOT NULL,
                profit        REAL    NOT NULL,
                open_positions INTEGER NOT NULL,
                daily_profit  REAL,
                weekly_profit REAL
            );
            """
        )

        # ----------------------------------------------------------------
        # Performance metrics – one row per day (can be extended)
        # ----------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT    NOT NULL UNIQUE,
                trades        INTEGER DEFAULT 0,
                wins          INTEGER DEFAULT 0,
                losses        INTEGER DEFAULT 0,
                win_rate      REAL    DEFAULT 0,
                total_profit  REAL    DEFAULT 0,
                avg_r_multiple REAL   DEFAULT 0,
                max_drawdown  REAL    DEFAULT 0,
                profit_factor REAL    DEFAULT 0
            );
            """
        )

        # ----------------------------------------------------------------
        # System events – audit trail for anything noteworthy
        # ----------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                event_type TEXT   NOT NULL,
                severity   TEXT   NOT NULL,
                message    TEXT   NOT NULL,
                details    TEXT
            );
            """
        )

        conn.commit()
        logger.info("✅ Database schema ensured")

    # ------------------------------------------------------------------
    # -------------------------- TRADE OPERATIONS ----------------------
    # ------------------------------------------------------------------
    def insert_trade(self, trade: TradeRecord) -> bool:
        """
        Insert a brand‑new trade.

        Returns ``True`` on success, ``False`` otherwise.
        """
        try:
            cur = self._connect().cursor()
            cur.execute(
                """
                INSERT INTO trades (
                    ticket, symbol, direction, entry_time, entry_price,
                    stop_loss, take_profit, lot_size, risk_amount, stack_level,
                    quality_score, confluence_score, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?, ?, ?);
                """,
                (
                    trade.ticket,
                    trade.symbol,
                    trade.direction,
                    trade.entry_time,
                    trade.entry_price,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.lot_size,
                    trade.risk_amount,
                    trade.stack_level,
                    trade.quality_score,
                    trade.confluence_score,
                    trade.notes,
                ),
            )
            logger.info(f"✅ Trade #{trade.ticket} recorded")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Trade #{trade.ticket} already exists – skipped")
            return False
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error inserting trade #{trade.ticket}: {exc}")
            return False

    # ------------------------------------------------------------------
    def update_trade_exit(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        r_multiple: float,
        exit_reason: Optional[str] = None,
    ) -> bool:
        """
        Record the exit details of a trade.

        Returns ``True`` on success.
        """
        try:
            cur = self._connect().cursor()
            cur.execute(
                """
                UPDATE trades
                SET exit_time = ?, exit_price = ?, profit = ?, r_multiple = ?, exit_reason = ?
                WHERE ticket = ?;
                """,
                (
                    datetime.now().isoformat(),
                    exit_price,
                    profit,
                    r_multiple,
                    exit_reason,
                    ticket,
                ),
            )
            logger.info(f"✅ Trade #{ticket} exit updated")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error updating exit for trade #{ticket}: {exc}")
            return False

    # ------------------------------------------------------------------
    def update_breakeven_flag(self, ticket: int) -> bool:
        """Mark a trade as having moved its breakeven point."""
        try:
            cur = self._connect().cursor()
            cur.execute(
                "UPDATE trades SET was_breakeven_moved = 1 WHERE ticket = ?;",
                (ticket,),
            )
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error setting breakeven flag for #{ticket}: {exc}")
            return False

    # ------------------------------------------------------------------
    def update_partial_flag(self, ticket: int) -> bool:
        """Mark a trade as partially closed."""
        try:
            cur = self._connect().cursor()
            cur.execute(
                "UPDATE trades SET was_partial_closed = 1 WHERE ticket = ?;",
                (ticket,),
            )
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error setting partial flag for #{ticket}: {exc}")
            return False

    # ------------------------------------------------------------------
    # ---------------------------- QUERIES -----------------------------
    # ------------------------------------------------------------------
    def get_trade(self, ticket: int) -> Optional[Dict]:
        """Fetch a single trade by its ticket number."""
        cur = self._connect().cursor()
        cur.execute("SELECT * FROM trades WHERE ticket = ?;", (ticket,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    def get_open_trades(self) -> List[Dict]:
        """All trades that have not yet been closed."""
        cur = self._connect().cursor()
        cur.execute("SELECT * FROM trades WHERE exit_time IS NULL;")
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    def get_closed_trades(self, limit: int = 100) -> List[Dict]:
        """Most recent closed trades (default 100)."""
        cur = self._connect().cursor()
        cur.execute(
            """
            SELECT * FROM trades
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?;
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    def get_trades_by_date_range(self, start_iso: str, end_iso: str) -> List[Dict]:
        """Trades whose entry_time falls between the two ISO timestamps."""
        cur = self._connect().cursor()
        cur.execute(
            """
            SELECT * FROM trades
            WHERE entry_time BETWEEN ? AND ?
            ORDER BY entry_time DESC;
            """,
            (start_iso, end_iso),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    def get_trades_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Recent trades for a particular symbol."""
        cur = self._connect().cursor()
        cur.execute(
            """
            SELECT * FROM trades
            WHERE symbol = ?
            ORDER BY entry_time DESC
            LIMIT ?;
            """,
            (symbol, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # --------------------------- ANALYTICS ---------------------------
    # ------------------------------------------------------------------
    def _empty_stats(self) -> Dict:
        """Template for an empty statistics payload."""
        return {
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_r_multiple": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "expectancy": 0.0,
        }

    # ------------------------------------------------------------------
    def calculate_statistics(self, days: int = 30) -> Dict:
        """
        Compute performance metrics for the last ``days`` days.

        Returns a dictionary with the most common KPIs.
        """
        cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat()
        cur = self._connect().cursor()
        cur.execute(
            """
            SELECT * FROM trades
            WHERE exit_time IS NOT NULL
              AND exit_time >= ?;
            """,
            (cutoff_iso,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return self._empty_stats()

        total = len(rows)
        winners = [r for r in rows if r["profit"] and r["profit"] > 0]
        losers = [r for r in rows if r["profit"] and r["profit"] <= 0]

        win_rate = (len(winners) / total) * 100 if total else 0.0
        total_profit = sum(r["profit"] for r in rows if r["profit"])
        total_wins = sum(r["profit"] for r in winners) if winners else 0.0
        total_losses = abs(
            sum(r["profit"] for r in losers) if losers else 0.0
        )
        avg_win = total_wins / len(winners) if winners else 0.0
        avg_loss = total_losses / len(losers) if losers else 0.0
        profit_factor = (total_wins / total_losses) if total_losses else 0.0
        avg_r = (
            sum(r["r_multiple"] for r in rows if r["r_multiple"]) / total
        )
        best_trade = max(rows, key=lambda x: x["profit"] or 0)["profit"]
        worst_trade = min(rows, key=lambda x: x["profit"] or 0)["profit"]
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        return {
            "period_days": days,
            "total_trades": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_r_multiple": avg_r,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "expectancy": expectancy,
        }

    # ------------------------------------------------------------------
    # ----------------------- ACCOUNT SNAPSHOTS -----------------------
    # ------------------------------------------------------------------
    def record_snapshot(
        self,
        balance: float,
        equity: float,
        margin_free: float,
        profit: float,
        open_positions: int,
        daily_profit: Optional[float] = None,
        weekly_profit: Optional[float] = None,
    ) -> bool:
        """Persist a point‑in‑time snapshot of the MT5 account."""
        try:
            cur = self._connect().cursor()
            cur.execute(
                """
                INSERT INTO account_snapshots (
                    timestamp, balance, equity, margin_free,
                    profit, open_positions, daily_profit, weekly_profit
                ) VALUES (?,?,?,?,?,?,?,?);
                """,
                (
                    datetime.now().isoformat(),
                    balance,
                    equity,
                    margin_free,
                    profit,
                    open_positions,
                    daily_profit,
                    weekly_profit,
                ),
            )
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error recording account snapshot: {exc}")
            return False

    # ------------------------------------------------------------------
    # --------------------------- EXPORTS ----------------------------
    # ------------------------------------------------------------------
    def export_to_json(self, output_path: str = "trades_export.json") -> bool:
        """Dump the entire `trades` table to a JSON file."""
        try:
            cur = self._connect().cursor()
            cur.execute("SELECT * FROM trades ORDER BY entry_time DESC;")
            rows = [dict(r) for r in cur.fetchall()]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, default=str)
            logger.info(f"✅ Exported {len(rows)} trades → {output_path}")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error exporting to JSON: {exc}")
            return False

    # ------------------------------------------------------------------
    def export_to_csv(self, output_path: str = "trades_export.csv") -> bool:
        """Dump the entire `trades` table to a CSV file."""
        try:
            cur = self._connect().cursor()
            cur.execute("SELECT * FROM trades ORDER BY entry_time DESC;")
            rows = [dict(r) for r in cur.fetchall()]
            if not rows:
                logger.warning("No trades to export")
                return False
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"✅ Exported {len(rows)} trades → {output_path}")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error exporting to CSV: {exc}")
            return False

    # ------------------------------------------------------------------
    # -------------------------- SYSTEM LOG --------------------------
    # ------------------------------------------------------------------
    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: Optional[str] = None,
    ) -> bool:
        """
        Record a generic system event (audit trail).

        Args:
            event_type: e.g. "ERROR", "INFO", "TRADE"
            severity:   "LOW", "MEDIUM", "HIGH"
            message:    Human‑readable description
            details:    Optional JSON / free‑form text
        """
        try:
            cur = self._connect().cursor()
            cur.execute(
                """
                              INSERT INTO system_events (
                    timestamp, event_type, severity, message, details
                ) VALUES (?,?,?,?,?);
                """,
                (
                    datetime.utcnow().isoformat(),
                    event_type,
                    severity,
                    message,
                    details,
                ),
            )
            self.conn.commit()
            logger.info(f"🛈 Event logged – {event_type}/{severity}: {message}")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"Error logging system event: {exc}")
            return False

    # ------------------------------------------------------------------
    # CLEAN‑UP
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the SQLite connection – should be called on shutdown."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("📊 Trade database connection closed")

# ----------------------------------------------------------------------
# GLOBAL SINGLETON – importable from anywhere in the code‑base
# ----------------------------------------------------------------------
trade_db = TradeDatabase()

# ----------------------------------------------------------------------
# QUICK DEMO / SELF‑TEST (run this file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import random

    print("\n" + "=" * 80)
    print("📊 TRADE DATABASE QUICK DEMO")
    print("=" * 80 + "\n")

    # Use an in‑memory DB for the demo so we don't touch the real file
    demo_db = TradeDatabase(":memory:")

    # ------------------------------------------------------------------
    # 1️⃣  Insert a few dummy trades
    # ------------------------------------------------------------------
    for i in range(1, 6):
        tr = TradeRecord(
            ticket=1000 + i,
            symbol=random.choice(["EURUSD", "GBPUSD", "USDJPY"]),
            direction=random.choice(["BUY", "SELL"]),
            entry_time=datetime.utcnow().isoformat(),
            entry_price=round(random.uniform(1.0, 1.5), 5),
            stop_loss=round(random.uniform(0.9, 1.0), 5),
            take_profit=round(random.uniform(1.5, 2.0), 5),
            lot_size=0.1,
            risk_amount=100,
            stack_level=1,
            quality_score=round(random.uniform(70, 95), 2),
            confluence_score=random.randint(3, 5),
            notes="Demo trade",
        )
        demo_db.insert_trade(tr)

    # ------------------------------------------------------------------
    # 2️⃣  Close a couple of trades
    # ------------------------------------------------------------------
    open_trades = demo_db.get_open_trades()
    for ot in open_trades[:2]:
        exit_price = ot["entry_price"] * (1.02 if ot["direction"] == "BUY" else 0.98)
        profit = (exit_price - ot["entry_price"]) * ot["lot_size"] * 100000
        r_multi = profit / ot["risk_amount"]
        demo_db.update_trade_exit(
            ticket=ot["ticket"],
            exit_price=exit_price,
            profit=profit,
            r_multiple=r_multi,
            exit_reason="Demo close",
        )

    # ------------------------------------------------------------------
    # 3️⃣  Show some analytics
    # ------------------------------------------------------------------
    stats = demo_db.calculate_statistics(days=365)
    print("\n📈 Sample statistics (last year):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k:20}: {v:,.2f}")
        else:
            print(f"   {k:20}: {v}")

    # ------------------------------------------------------------------
    # 4️⃣  Export to JSON / CSV (to the current folder)
    # ------------------------------------------------------------------
    demo_db.export_to_json("demo_trades.json")
    demo_db.export_to_csv("demo_trades.csv")

    # ------------------------------------------------------------------
    # 5️⃣  Log a system event
    # ------------------------------------------------------------------
    demo_db.log_event(
        event_type="INFO",
        severity="LOW",
        message="Demo run completed",
        details=json.dumps({"trades_inserted": 5, "trades_closed": 2}),
    )

    # ------------------------------------------------------------------
    # 6️⃣  Clean‑up
    # ------------------------------------------------------------------
    demo_db.close()
    print("\n✅ Demo finished – check the generated JSON/CSV files.\n")
    print("=" * 80 + "\n")
