#!/usr/bin/env python3


Automated trade history backup and export system.
Integrates with existing SQLite database and MT5 CSV exports.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import os
import shutil
import sqlite3
import json
import zipfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import pandas as pd

# ----------------------------------------------------------------------
# Logging configuration (expects the application to configure the root logger)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# TradeHistoryBackup – the core class
# ----------------------------------------------------------------------
class TradeHistoryBackup:
    """
    Comprehensive trade‑history backup system.

    Features
    --------
    * Automated SQLite database backups
    * MT5 CSV import → SQLite
    * Export to CSV, JSON, Excel (with optional analytics sheet)
    * ZIP package for cloud storage (DB + all exports + metadata)
    * Cleanup of old backups
    * Summary of existing backup files
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        db_path: str = "trading_history.db",
        backup_dir: str = "backups",
    ) -> None:
        """
        Initialise the backup manager.

        Parameters
        ----------
        db_path: Path to the SQLite database.
        backup_dir: Directory where all backup artefacts will be stored.
        """
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.mt5_files_path = self._find_mt5_files_path()

        # Ensure the backup directory exists
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)

        logger.info("📦 Trade History Backup Manager initialised")
        logger.info(f"   Database      : {self.db_path}")
        logger.info(f"   Backup folder : {self.backup_dir}")

    # ------------------------------------------------------------------
    # Helper – locate MT5 Files folder (used for CSV import/export)
    # ------------------------------------------------------------------
    def _find_mt5_files_path(self) -> Optional[str]:
        """Attempt to locate the MT5 *Files* directory."""
        possible_paths = [
            os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal"),
            r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal",
            r"C:\Program Files\MetaTrader 5\MQL5\Files",
        ]

        for base_path in possible_paths:
            if not os.path.isdir(base_path):
                continue

            # Terminal directories contain sub‑folders for each installation
            if "Terminal" in base_path:
                for folder in os.listdir(base_path):
                    candidate = os.path.join(base_path, folder, "MQL5", "Files")
                    if os.path.isdir(candidate):
                        return candidate
            else:
                return base_path

        logger.warning("MT5 Files directory not found – CSV import/export will be limited")
        return None

    # ------------------------------------------------------------------
    # 1️⃣  Database backup
    # ------------------------------------------------------------------
    def backup_database(self, include_timestamp: bool = True) -> Optional[str]:
        """
        Create a copy of the SQLite database.

        Parameters
        ----------
        include_timestamp: If True, the backup filename contains a timestamp.

        Returns
        -------
        Path to the backup file, or None on failure.
        """
        if not os.path.isfile(self.db_path):
            logger.error(f"❌ Database not found at {self.db_path}")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            suffix = f"_{timestamp}" if timestamp else ""
            backup_name = f"trading_history_backup{suffix}.db"
            backup_path = os.path.join(self.backup_dir, backup_name)

            shutil.copy2(self.db_path, backup_path)
            size_kb = os.path.getsize(backup_path) / 1024
            logger.info(f"✅ Database backup created: {backup_path} ({size_kb:.2f} KB)")
            return backup_path
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ Failed to backup database: {exc}")
            return None

    # ------------------------------------------------------------------
    # 2️⃣  Import MT5 CSV export into SQLite
    # ------------------------------------------------------------------
    def import_mt5_csv(self, csv_path: Optional[str] = None) -> bool:
        """
        Import a CSV file exported from MT5 into the SQLite DB.

        Parameters
        ----------
        csv_path: Path to the CSV file. If omitted, the method attempts to locate
                  a default file in the MT5 *Files* folder.

        Returns
        -------
        True on success, False otherwise.
        """
        # Locate CSV if not supplied
        if csv_path is None:
            csv_path = self._find_mt5_csv()

        if not csv_path or not os.path.isfile(csv_path):
            logger.error(f"❌ CSV file not found: {csv_path}")
            return False

        try:
            df = pd.read_csv(csv_path)
            logger.info(f"📥 Loaded {len(df)} rows from {csv_path}")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            imported = 0
            skipped = 0

            for _, row in df.iterrows():
                # Check for duplicate ticket
                cursor.execute("SELECT ticket FROM trades WHERE ticket = ?", (row["ticket"],))
                if cursor.fetchone():
                    skipped += 1
                    continue

                # Insert – map CSV columns to DB schema
                try:
                    cursor.execute(
                        """
                        INSERT INTO trades (
                            ticket, symbol, direction, entry_time, entry_price,
                            stop_loss, take_profit, lot_size, risk_amount, stack_level,
                            exit_time, exit_price, profit, notes
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            int(row["ticket"]),
                            row["symbol"],
                            row["direction"],
                            row["entry_time"],
                            float(row["entry_price"]),
                            float(row.get("stop_loss", 0)),
                            float(row.get("take_profit", 0)),
                            float(row["volume"]),
                            0,               # risk_amount not present in CSV
                            1,               # default stack level
                            row.get("exit_time"),
                            float(row.get("exit_price", 0))
                            if pd.notna(row.get("exit_price"))
                            else None,
                            float(row.get("profit", 0)),
                            row.get("comment", ""),
                        ),
                    )
                    imported += 1
                except Exception as e:  # pragma: no cover
                    logger.warning(f"⚠️ Failed to insert ticket {row['ticket']}: {e}")
                    skipped += 1

            conn.commit()
            conn.close()

            logger.info(f"✅ CSV import finished – {imported} new, {skipped} skipped")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ CSV import failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Helper – locate the default MT5 CSV export
    # ------------------------------------------------------------------
    def _find_mt5_csv(self) -> Optional[str]:
        """Search for a CSV file named ``trade_history_backup.csv``."""
        if self.mt5_files_path:
            candidate = os.path.join(self.mt5_files_path, "trade_history_backup.csv")
            if os.path.isfile(candidate):
                return candidate

        # Fallback to current working directory
        cwd_candidate = "trade_history_backup.csv"
        if os.path.isfile(cwd_candidate):
            return cwd_candidate

        return None

    # ------------------------------------------------------------------
    # 3️⃣  Export to CSV
    # ------------------------------------------------------------------
    def export_to_csv(self, output_path: Optional[str] = None, days: Optional[int] = None) -> Optional[str]:
        """
        Export trades from the SQLite DB to a CSV file.

        Parameters
        ----------
        output_path: Destination file. If omitted, a timestamped file is created.
        days: If supplied, only trades from the last *days* are exported.

        Returns
        -------
        Path to the CSV file, or None on failure.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.backup_dir, f"trades_export_{ts}.csv")

        try:
            conn = sqlite3.connect(self.db_path)
            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                query = f"SELECT * FROM trades WHERE entry_time >= '{cutoff}' ORDER BY entry_time DESC"
            else:
                query = "SELECT * FROM trades ORDER BY entry_time DESC"

            df = pd.read_sql_query(query, conn)
            conn.close()

            df.to_csv(output_path, index=False)
            logger.info(f"✅ Exported {len(df)} rows to CSV: {output_path}")
            return output_path
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ CSV export failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # 4️⃣  Export to Excel (multiple sheets + optional analytics)
    # ------------------------------------------------------------------
    def export_to_excel(
        self,
        output_path: Optional[str] = None,
        include_analytics: bool = True,
    ) -> Optional[str]:
        """
        Export trades to an Excel workbook with several sheets.

        Parameters
        ----------
        output_path: Destination file. If omitted, a timestamped file is created.
        include_analytics: If True, an additional sheet with performance metrics is added.

        Returns
        -------
        Path to the Excel file, or None on failure.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.backup_dir, f"trades_export_{ts}.xlsx")

        try:
            conn = sqlite3.connect(self.db_path)

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # All trades
                df_all = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time DESC", conn)
                df_all.to_excel(writer, sheet_name="All Trades", index=False)

                # Open positions
                df_open = pd.read_sql_query(
                    "SELECT * FROM trades WHERE exit_time IS NULL ORDER BY entry_time DESC", conn
                )
                df_open.to_excel(writer, sheet_name="Open Positions", index=False)

                # Closed trades
                df_closed = pd.read_sql_query(
                    "SELECT * FROM trades WHERE exit_time IS NOT NULL ORDER BY exit_time DESC", conn
                )
                df_closed.to_excel(writer, sheet_name="Closed Trades", index=False)

                # Analytics sheet (optional)
                if include_analytics:
                    analytics = self._generate_analytics(conn)
                    df_analytics = pd.DataFrame([analytics])
                    df_analytics.to_excel(writer, sheet_name="Analytics", index=False)

            conn.close()
            logger.info(f"✅ Excel workbook created: {output_path}")
            return output_path
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ Excel export failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal – compute simple performance analytics
    # ------------------------------------------------------------------
    def _generate_analytics(self, conn: sqlite3.Connection) -> Dict:
        """Return a dictionary with basic performance metrics."""
        cur = conn.cursor()
        cur.execute("SELECT * FROM trades WHERE exit_time IS NOT NULL")
        rows = cur.fetchall()

        if not rows:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_profit": 0,
            }

        total = len(rows)
        wins = [r[12] for r in rows if r[12] and r[12] > 0]   # profit column
        losses = [r[12] for r in rows if r[12] and r[12] <= 0]

        win_rate = len(wins) / total * 100
        total_profit = sum(wins) + sum(losses)  # losses are negative
        profit_factor = (sum(wins) / -sum(losses)) if losses else 0

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "total_profit": round(total_profit, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(-sum(losses) / len(losses), 2) if losses else 0,
        }

    # ------------------------------------------------------------------
    # 5️⃣  Export to JSON
    # ------------------------------------------------------------------
    def export_to_json(self, output_path: Optional[str] = None, pretty: bool = True) -> Optional[str]:
        """
        Export all trades to a JSON file.

        Parameters
        ----------
        output_path: Destination file. If omitted, a timestamped file is created.
        pretty: If True, the JSON is indented for readability.

        Returns
        -------
        Path to the JSON file, or None on failure.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.backup_dir, f"trades_export_{ts}.json")

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM trades ORDER BY entry_time DESC")
            trades = [dict(row) for row in cur.fetchall()]
            conn.close()

            with open(output_path, "w") as f:
                if pretty:
                    json.dump(trades, f, indent=2, default=str)
                else:
                    json.dump(trades, f, default=str)

            logger.info(f"✅ JSON export created: {output_path}")
            return output_path
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ JSON export failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # 6️⃣  Create a cloud‑ready ZIP package (DB + all exports + metadata)
    # ------------------------------------------------------------------
    def create_cloud_backup_package(self, include_analytics: bool = True) -> Optional[str]:
        """
        Assemble a single ZIP file containing:
        * SQLite backup
        * CSV export
        * JSON export
        * Excel workbook (with optional analytics)
        * Metadata JSON describing the package

        Returns
        -------
        Path to the ZIP file, or None on failure.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(self.backup_dir, f"complete_backup_{ts}.zip")

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # 1️⃣ DB backup (no timestamp – we add our own)
                db_backup = self.backup_database(include_timestamp=False)
                if db_backup:
                    zf.write(db_backup, os.path.basename(db_backup))

                # 2️⃣ CSV export
                csv_path = self.export_to_csv()
                if csv_path:
                    zf.write(csv_path, os.path.basename(csv_path))

                # 3️⃣ JSON export
                json_path = self.export_to_json()
                if json_path:
                    zf.write(json_path, os.path.basename(json_path))

                # 4️⃣ Excel (with analytics if requested)
                excel_path = self.export_to_excel(include_analytics=include_analytics)
                if excel_path:
                    zf.write(excel_path, os.path.basename(excel_path))

                # 5️⃣ Metadata file
                meta = {
                    "created": datetime.now().isoformat(),
                    "database": self.db_path,
                    "included_files": [
                        "SQLite backup (.db)",
                        "CSV export",
                        "JSON export",
                        "Excel workbook",
                    ],
                    "analytics_included": include_analytics,
                }
                zf.writestr("backup_metadata.json", json.dumps(meta, indent=2))

            size_kb = os.path.getsize(zip_path) / 1024
            logger.info(f"✅ Cloud backup package created: {zip_path} ({size_kb:.2f} KB)")
            return zip_path
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ Cloud backup creation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # 7️⃣  Cleanup old backup artefacts
    # ------------------------------------------------------------------
    def cleanup_old_backups(self, days: int = 30) -> int:
        """
        Delete backup files older than *days*.

        Parameters
        ----------
        days: Age threshold in days.

        Returns
        -------
        Number of files removed.
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        try:
            for fname in os.listdir(self.backup_dir):
                fpath = os.path.join(self.backup_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                mod_time = datetime.fromtimestamp(os.path.getmtime(fpath))
                if mod_time < cutoff:
                    os.remove(fpath)
                    removed += 1
                    logger.info(f"🗑️ Removed old backup: {fname}")

            logger.info(f"✅ Cleanup finished – {removed} files deleted")
            return removed
        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ Cleanup failed: {exc}")
            return 0

    # ------------------------------------------------------------------
    # 8️⃣  Summary of existing backups
    # ------------------------------------------------------------------
    def get_backup_summary(self) -> Dict:
        """
        Scan the backup directory and return a summary dictionary.

        Returns
        -------
        {
            "total_backups": int,
            "total_size_mb": float,
            "backups": [
                {
                    "filename": str,
                    "size_kb": float,
                    "modified": str,      # YYYY‑MM‑DD HH:MM:SS
                    "age_days": int
                },
                …
            ]
        }
        """
        backups: List[Dict] = []
        total_bytes = 0

        try:
            for fname in os.listdir(self.backup_dir):
                fpath = os.path.join(self.backup_dir, fname)

                if not os.path.isfile(fpath):
                    continue

                size = os.path.getsize(fpath)
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                age_days = (datetime.now() - mtime).days

                backups.append(
                    {
                        "filename": fname,
                        "size_kb": size / 1024,
                        "modified": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                        "age_days": age_days,
                    }
                )
                total_bytes += size

            summary = {
                "total_backups": len(backups),
                "total_size_mb": total_bytes / (1024 * 1024),
                "backups": sorted(
                    backups, key=lambda x: x["modified"], reverse=True
                ),
            }
            return summary

        except Exception as exc:  # pragma: no cover
            logger.error(f"❌ Failed to generate backup summary: {exc}")
            return {"total_backups": 0, "total_size_mb": 0.0, "backups": []}


# ----------------------------------------------------------------------
# Simple CLI test harness – run this file directly to see everything in action
# ----------------------------------------------------------------------
def main() -> None:
    """Demonstrate the full feature set of TradeHistoryBackup."""
    print("═" * 70)
    print("      TRADE HISTORY BACKUP MANAGER – DEMO")
    print("═" * 70)
    print()

    # ------------------------------------------------------------------
    # Initialise manager (uses defaults: trading_history.db & backups/)
    # ------------------------------------------------------------------
    backup_mgr = TradeHistoryBackup()

    # ------------------------------------------------------------------
    # 1️⃣  Database backup
    # ------------------------------------------------------------------
    print("📦 Test 1 – Database backup")
    print("─" * 70)
    db_file = backup_mgr.backup_database()
    if db_file:
        print(f"✅ DB backup created: {db_file}")
    print()

    # ------------------------------------------------------------------
    # 2️⃣  CSV export
    # ------------------------------------------------------------------
    print("📊 Test 2 – Export to CSV")
    print("─" * 70)
    csv_file = backup_mgr.export_to_csv()
    if csv_file:
        print(f"✅ CSV exported: {csv_file}")
    print()

    # ------------------------------------------------------------------
    # 3️⃣  JSON export
    # ------------------------------------------------------------------
    print("📊 Test 3 – Export to JSON")
    print("─" * 70)
    json_file = backup_mgr.export_to_json()
    if json_file:
        print(f"✅ JSON exported: {json_file}")
    print()

    # ------------------------------------------------------------------
    # 4️⃣  Excel export (with analytics)
    # ------------------------------------------------------------------
    print("📊 Test 4 – Export to Excel (analytics)")
    print("─" * 70)
    excel_file = backup_mgr.export_to_excel(include_analytics=True)
    if excel_file:
        print(f"✅ Excel exported: {excel_file}")
    print()

    # ------------------------------------------------------------------
    # 5️⃣  Cloud‑ready ZIP package
    # ------------------------------------------------------------------
    print("☁️  Test 5 – Create cloud backup package")
    print("─" * 70)
    zip_file = backup_mgr.create_cloud_backup_package()
    if zip_file:
        print(f"✅ ZIP package created: {zip_file}")
    print()

    # ------------------------------------------------------------------
    # 6️⃣  Backup summary
    # ------------------------------------------------------------------
    print("📋 Test 6 – Backup summary")
    print("─" * 70)
    summary = backup_mgr.get_backup_summary()
    print(f"Total backups : {summary['total_backups']}")
    print(f"Total size    : {summary['total_size_mb']:.2f} MB")
    print("\nRecent backups:")
    for b in summary["backups"][:5]:
        print(
            f"  • {b['filename']} – {b['size_kb']:.2f} KB – "
            f"{b['modified']} ({b['age_days']} d ago)"
        )
    print()

    # ------------------------------------------------------------------
    # 7️⃣  Cleanup old backups (example: delete > 60 days)
    # ------------------------------------------------------------------
    print("🧹 Test 7 – Cleanup old backups (>60 days)")
    print("─" * 70)
    removed = backup_mgr.cleanup_old_backups(days=60)
    print(f"✅ Removed {removed} old backup file(s)")
    print()

    print("═" * 70)
    print("✅ Demo completed – all functions exercised")
    print("═" * 70)


if __name__ == "__main__":
    # Simple console logging for the demo
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
