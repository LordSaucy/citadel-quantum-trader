!/usr/bin/env python3
"""
DASHBOARD CONNECTOR

Bridges the Mission Control Dashboard with the trading bot.
Handles real‑time data updates and configuration synchronization.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production‑Ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import threading
import time
from datetime import datetime, date, time as dt_time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ----------------------------------------------------------------------
# Logging (the global CQT logging config will already be active;
# we just obtain a module‑level logger)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# DashboardConnector implementation
# ----------------------------------------------------------------------
class DashboardConnector:
    """
    Connects the Citadel Quantum Trader to the Mission‑Control Dashboard.

    * Loads a JSON configuration file (`mission_control_config.json`).
    * Watches the file for external edits and notifies the bot via callbacks.
    * Pushes live data snapshots to `mission_control_data.json`.
    * Exposes a rich set of convenience getters so the rest of the code
      never needs to touch the raw dict.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config_file: str = "mission_control_config.json"):
        """
        Initialise the connector.

        Args:
            config_file: Path (relative to the CQT working directory) of the
                         JSON file that the dashboard edits.
        """
        # ---- file locations -------------------------------------------------
        self.config_path: Path = Path(config_file)
        self.data_path: Path = Path("mission_control_data.json")

        # ---- internal state -------------------------------------------------
        self.config: Dict = self._load_config()
        self._last_mtime: float = self._config_mtime()
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[Callable[[Dict], None]] = []

        # ---- start background watcher ---------------------------------------
        self._start_watcher()
        logger.info("DashboardConnector initialised (watching %s)", self.config_path)

    # ------------------------------------------------------------------
    # 1️⃣  Config handling
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict:
        """Read the JSON config file; fall back to defaults on error."""
        if self.config_path.is_file():
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                logger.info("Loaded dashboard configuration from %s", self.config_path)
                return cfg
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to parse %s: %s", self.config_path, exc)

        logger.warning("Using default dashboard configuration")
        return self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        """Hard‑coded sane defaults – identical to the example you supplied."""
        return {
            "stacking_enabled": True,
            "rr_1_2_enabled": True,
            "rr_1_4_enabled": False,
            "breakeven_enabled": True,
            "partial_close_enabled": True,
            "trailing_stop_enabled": False,
            "news_filter_enabled": True,
            "session_filter_enabled": True,
            "mtf_confirmation_enabled": True,
            "confluence_filter_enabled": True,
            "max_positions": 4,
            "max_stack_level": 4,
            "daily_loss_limit": 5_000,
            "max_drawdown_pct": 10.0,
            "risk_per_trade_pct": 2.0,
            "min_entry_quality": 85,
            "min_confluence_score": 4,
            "mtf_alignment_threshold": 70,
            "breakeven_trigger_r": 1.0,
            "partial_close_r": 2.0,
            "partial_close_pct": 50,
            "trailing_start_r": 2.0,
            "trailing_distance_r": 0.5,
            "platforms_enabled": {
                "MT5": True,
                "IB": False,
                "cTrader": False,
                "NinjaTrader": False,
                "Tradovate": False,
            },
        }

    def _config_mtime(self) -> float:
        """Return the last‑modification timestamp of the config file (0 if missing)."""
        return self.config_path.stat().st_mtime if self.config_path.is_file() else 0.0

    # ------------------------------------------------------------------
    # 2️⃣  Hot‑reload watcher (daemon thread)
    # ------------------------------------------------------------------
    def _start_watcher(self) -> None:
        """Launch a background thread that polls the config file every 2 s."""
        def _watch():
            logger.debug("Dashboard config watcher thread started")
            while not self._stop_event.is_set():
                try:
                    cur_mtime = self._config_mtime()
                    if cur_mtime > self._last_mtime:
                        logger.info("Dashboard configuration changed – reloading")
                        self.config = self._load_config()
                        self._last_mtime = cur_mtime
                        # fire all registered callbacks
                        for cb in self._callbacks:
                            try:
                                cb(self.config)
                            except Exception as exc:  # pragma: no cover
                                logger.error("Config callback raised: %s", exc)
                    time.sleep(2.0)
                except Exception as exc:  # pragma: no cover
                    logger.error("Error in config watcher: %s", exc)
                    time.sleep(5.0)
            logger.debug("Dashboard config watcher thread exiting")

        self._watcher_thread = threading.Thread(target=_watch, daemon=True)
        self._watcher_thread.start()

    def register_config_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Register a function that will be called **once** every time the
        dashboard configuration file is re‑loaded.

        The callback receives the *new* configuration dict as its sole argument.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callbacks.append(callback)
        logger.debug("Registered config‑change callback %s", callback)

    # ------------------------------------------------------------------
    # 3️⃣  Real‑time data push
    # ------------------------------------------------------------------
    def update_dashboard_data(self, data: Dict) -> None:
        """
        Write a JSON snapshot that the Mission‑Control UI can poll.

        The method adds a ``timestamp`` field (ISO‑8601 UTC) and writes the
        file atomically (write‑to‑tmp → rename) to avoid readers seeing a
        partially‑written file.
        """
        payload = dict(data)  # shallow copy
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

        tmp_path = self.data_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
            tmp_path.replace(self.data_path)  # atomic rename on POSIX
            logger.debug("Dashboard data updated (%s bytes)", len(json.dumps(payload)))
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to write dashboard data: %s", exc)

    # ------------------------------------------------------------------
    # 4️⃣  Convenience getters – hide the raw dict behind typed methods
    # ------------------------------------------------------------------
    # Risk‑reward helpers -------------------------------------------------
    def get_risk_reward_ratio(self) -> float:
        """Return the active R:R multiplier (4 : 1, 2 : 1 or 1 : 1)."""
        if self.config.get("rr_1_4_enabled"):
            return 4.0
        if self.config.get("rr_1_2_enabled", True):
            return 2.0
        return 1.0

    # Stacking ------------------------------------------------------------
    def should_use_stacking(self) -> bool:
        return bool(self.config.get("stacking_enabled", True))

    def get_max_stack_level(self) -> int:
        return int(self.config.get("max_stack_level", 4))

    # Breakeven -----------------------------------------------------------
    def should_move_to_breakeven(self) -> bool:
        return bool(self.config.get("breakeven_enabled", True))

    def get_breakeven_trigger(self) -> float:
        return float(self.config.get("breakeven_trigger_r", 1.0))

    # Partial‑close -------------------------------------------------------
    def should_partial_close(self) -> bool:
        return bool(self.config.get("partial_close_enabled", True))

    def get_partial_close_params(self) -> Dict:
        return {
            "trigger_r": float(self.config.get("partial_close_r", 2.0)),
            "close_pct": int(self.config.get("partial_close_pct", 50)),
        }

    # Trailing‑stop -------------------------------------------------------
    def should_use_trailing_stop(self) -> bool:
        return bool(self.config.get("trailing_stop_enabled", False))

    def get_trailing_stop_params(self) -> Dict:
        return {
            "start_r": float(self.config.get("trailing_start_r", 2.0)),
            "distance_r": float(self.config.get("trailing_distance_r", 0.5)),
        }

    # Filters -------------------------------------------------------------
    def should_apply_news_filter(self) -> bool:
        return bool(self.config.get("news_filter_enabled", True))

    def should_apply_session_filter(self) -> bool:
        return bool(self.config.get("session_filter_enabled", True))

    def should_check_mtf_confirmation(self) -> bool:
        return bool(self.config.get("mtf_confirmation_enabled", True))

    def should_check_confluence(self) -> bool:
        return bool(self.config.get("confluence_filter_enabled", True))

    # Entry quality -------------------------------------------------------
    def get_entry_requirements(self) -> Dict:
        return {
            "min_quality": int(self.config.get("min_entry_quality", 85)),
            "min_confluence": int(self.config.get("min_confluence_score", 4)),
            "mtf_threshold": int(self.config.get("mtf_alignment_threshold", 70)),
        }

    # Risk limits ---------------------------------------------------------
    def get_risk_limits(self) -> Dict:
        return {
            "max_positions": int(self.config.get("max_positions", 4)),
            "daily_loss_limit": float(self.config.get("daily_loss_limit", 5_000)),
            "max_drawdown_pct": float(self.config.get("max_drawdown_pct", 10.0)),
            "risk_per_trade_pct": float(self.config.get("risk_per_trade_pct", 2.0)),
        }

    # Platform enable‑flags -----------------------------------------------
    def get_enabled_platforms(self) -> List[str]:
        platforms = self.config.get("platforms_enabled", {})
        return [p for p, enabled in platforms.items() if enabled]

    # ------------------------------------------------------------------
    # 5️⃣  Shutdown / cleanup
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Terminate the watcher thread and persist the latest config."""
        self._stop_event.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5)
        # Persist the *current* config so a fresh start picks it up
        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            logger.info("DashboardConnector stopped and config persisted")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to persist config on shutdown: %s", exc)


# ----------------------------------------------------------------------
# Global singleton – import this from anywhere in the code base
# ----------------------------------------------------------------------

