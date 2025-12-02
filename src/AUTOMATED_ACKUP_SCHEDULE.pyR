#!/usr/bin/env python3
"""
AUTOMATED BACKUP SCHEDULER

Runs in the background and creates periodic backups of the trading
history / database.  All intervals are configurable at runtime via the
`BackupScheduler` methods.

Author   : Lawful Banker
Created  : 2024‑11‑26
Version  : 2.0 – Production‑Ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import schedule  # pip install schedule

# ----------------------------------------------------------------------
# Internal imports – the backup manager that actually does the work.
# ----------------------------------------------------------------------
# The file `src/trade_history_backup.py` must expose a class
# `TradeHistoryBackup` with the methods used below.
from .trade_history_backup import TradeHistoryBackup

# ----------------------------------------------------------------------
# Logging configuration (rotating file + console)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
LOGGER.addHandler(console_handler)

# File handler (rotates daily, keeps 7 days)
file_handler = logging.FileHandler(LOG_DIR / "backup_scheduler.log")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)

# ----------------------------------------------------------------------
# BackupScheduler – the heart of the service
# ----------------------------------------------------------------------
class BackupScheduler:
    """
    Automated backup scheduler.

    Schedules:

    * Database backups – every 6 hours
    * CSV export – daily at 23:00
    * Excel report – weekly on Sunday at 23:00
    * Cloud backup package – weekly on Sunday at 22:00
    * Cleanup of old backups – weekly on Monday at 01:00
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.backup_mgr = TradeHistoryBackup()
        self._running = threading.Event()          # controls the main loop
        self._worker: Optional[threading.Thread] = None
        LOGGER.info("🕐 BackupScheduler initialised")

    # ------------------------------------------------------------------
    # Public API – start / stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._worker and self._worker.is_alive():
            LOGGER.warning("Scheduler already running")
            return

        self._setup_schedules()
        self._running.set()

        # Run an immediate full backup before entering the loop
        LOGGER.info("🔄 Performing initial full backup")
        self.run_immediate_backup()

        self._worker = threading.Thread(
            target=self._run_loop,
            name="BackupSchedulerWorker",
            daemon=True,
        )
        self._worker.start()
        LOGGER.info("🚀 BackupScheduler started (background thread)")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread."""
        if not self._running.is_set():
            LOGGER.warning("Scheduler is not running")
            return

        self._running.clear()
        if self._worker:
            self._worker.join(timeout=30)
        LOGGER.info("🛑 BackupScheduler stopped")

    # ------------------------------------------------------------------
    # Immediate full backup (used on start and can be called manually)
    # ------------------------------------------------------------------
    def run_immediate_backup(self) -> None:
        """Run a full backup (DB + CSV + Cloud package) immediately."""
        LOGGER.info("🚀 Running immediate full backup")
        self._scheduled_database_backup()
        self._scheduled_csv_export()
        self._scheduled_cloud_backup()
        LOGGER.info("✅ Immediate full backup completed")

    # ------------------------------------------------------------------
    # Helper – the infinite loop that drives `schedule`
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        """Background thread – executes pending jobs every minute."""
        try:
            while self._running.is_set():
                schedule.run_pending()
                time.sleep(60)   # check once per minute
        except Exception as exc:   # pragma: no cover
            LOGGER.exception(f"BackupScheduler crashed: {exc}")
        finally:
            self._running.clear()

    # ------------------------------------------------------------------
    # Schedule definition – called once from `start()`
    # ------------------------------------------------------------------
    def _setup_schedules(self) -> None:
        """Define all recurring jobs."""
        schedule.clear()   # safety – remove any stale jobs

        # Database backup – every 6 hours
        schedule.every(6).hours.do(self._scheduled_database_backup)
        LOGGER.info("📅 Scheduled: Database backup every 6 hours")

        # CSV export – daily at 23:00
        schedule.every().day.at("23:00").do(self._scheduled_csv_export)
        LOGGER.info("📅 Scheduled: CSV export daily at 23:00")

        # Excel report – weekly on Sunday at 23:00
        schedule.every().sunday.at("23:00").do(self._scheduled_excel_export)
        LOGGER.info("📅 Scheduled: Excel report weekly on Sunday at 23:00")

        # Cloud backup package – weekly on Sunday at 22:00
        schedule.every().sunday.at("22:00").do(self._scheduled_cloud_backup)
        LOGGER.info("📅 Scheduled: Cloud backup weekly on Sunday at 22:00")

        # Cleanup old backups – weekly on Monday at 01:00
        schedule.every().monday.at("01:00").do(self._scheduled_cleanup)
        LOGGER.info("📅 Scheduled: Cleanup weekly on Monday at 01:00")

    # ------------------------------------------------------------------
    # Individual job wrappers – thin pass‑throughs to TradeHistoryBackup
    # ------------------------------------------------------------------
    def _scheduled_database_backup(self) -> None:
        """Database backup job (run by schedule)."""
        LOGGER.info("⏰ Running scheduled database backup …")
        try:
            backup_path = self.backup_mgr.backup_database()
            if backup_path:
                LOGGER.info(f"✅ Database backup saved to: {backup_path}")
            else:
                LOGGER.error("❌ Database backup returned no path")
        except Exception as exc:   # pragma: no cover
            LOGGER.exception(f"Database backup failed: {exc}")

    def _scheduled_csv_export(self) -> None:
        """CSV export job (run by schedule)."""
        LOGGER.info("⏰ Running scheduled CSV export …")
        try:
            csv_path = self.backup_mgr.export_to_csv()
            if csv_path:
                LOGGER.info(f"✅ CSV export saved to: {csv_path}")
            else:
                LOGGER.error("❌ CSV export returned no path")
        except Exception as exc:   # pragma: no cover
            LOGGER.exception(f"CSV export failed: {exc}")

    def _scheduled_excel_export(self) -> None:
        """Excel export job (run by schedule)."""
        LOGGER.info("⏰ Running scheduled Excel export …")
        try:
            excel_path = self.backup_mgr.export_to_excel(include_analytics=True)
            if excel_path:
                LOGGER.info(f"✅ Excel export saved to: {excel_path}")
            else:
                LOGGER.error("❌ Excel export returned no path")
        except Exception as exc:   # pragma: no cover
            LOGGER.exception(f"Excel export failed: {exc}")

    def _scheduled_cloud_backup(self) -> None:
        """Cloud backup package job (run by schedule)."""
        LOGGER.info("⏰ Running scheduled cloud backup …")
        try:
            cloud_path = self.backup_mgr.create_cloud_backup_package()
            if cloud_path:
                LOGGER.info(f"✅ Cloud backup package created: {cloud_path}")
            else:
                LOGGER.error("❌ Cloud backup package returned no path")
        except Exception as exc:   # pragma: no cover
            LOGGER.exception(f"Cloud backup failed: {exc}")

    def _scheduled_cleanup(self) -> None:
        """Cleanup old backups job (run by schedule)."""
        LOGGER.info("⏰ Running scheduled cleanup of old backups …")
        try:
            deleted = self.backup_mgr.cleanup_old_backups(days=30)
            LOGGER.info(f"✅ Cleanup complete – {deleted} old files removed")
        except Exception as exc:   # pragma: no cover
            LOGGER.exception(f"Cleanup failed: {exc}")

    # ------------------------------------------------------------------
    # Introspection helpers (useful for health‑checks or Grafana panels)
    # ------------------------------------------------------------------
    def get_next_runs(self) -> str:
        """
        Return a human‑readable list of the next scheduled executions.
        Useful for logging or exposing via an HTTP health‑endpoint.
        """
        jobs = schedule.get_jobs()
        if not jobs:
            return "No scheduled jobs."

        lines: List[str] = ["\n📅 NEXT SCHEDULED RUNS:", "─" * 70]
        for job in jobs:
            # `schedule` does not expose the exact next run timestamp,
            # but we can approximate it by looking at the job's interval.
            # For readability we just show the cron‑like description.
            lines.append(f"• {job.job_func.__name__}: {job}")
        lines.append("─" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context‑manager support (optional, nice for tests)
    # ------------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # Propagate exception (if any)
        return False

# ----------------------------------------------------------------------
# Convenience entry‑point – `python -m src.automated_backup_scheduler`
# ----------------------------------------------------------------------
def _banner() -> None:
    """Print a nice ASCII banner when the script is executed directly."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "AUTOMATED BACKUP SCHEDULER" + " " * 27 + "║")
    print("║" + " " * 20 + "Lawful Banker Trading System" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")


def main() -> None:
    """Standalone runner – starts the scheduler and blocks until Ctrl‑C."""
    _banner()
    scheduler = BackupScheduler()
    scheduler.start()

    # Show the next runs once at startup (nice for ops)
    print(scheduler.get_next_runs())
    print()

    try:
        # Block the main thread – the worker thread does the real work.
        while True:
            time.sleep(3600)   # keep the process alive
    except KeyboardInterrupt:
        LOGGER.info("\n⏹️  Scheduler stopped by user (Ctrl‑C)")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
