#!/usr/bin/env python3
"""
Calendar Data Reader

Reads exported calendar data from the MT5 bridge (MQL5 script) and provides
a clean Python API that can be used by the trading engine, risk‑manager,
or any external service (Grafana, alerts, etc.).

If the bridge is not running the class falls back to the simplified
`MT5EconomicCalendar` implementation that queries MT5 directly.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import pandas as pd                     # pip install pandas
from flask import Flask, jsonify, request, abort  # pip install flask

# ----------------------------------------------------------------------
# Internal imports (same package)
# ----------------------------------------------------------------------
from .economic_calendar import (
    NewsEvent,
    NewsImpact,
    MT5EconomicCalendar,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 0️⃣  CalendarDataReader – reads the JSON file produced by the MQL5 bridge
# ----------------------------------------------------------------------
class CalendarDataReader:
    """
    Reads calendar data exported by the MT5 bridge.

    The bridge writes a file called ``calendar_data.json`` (and optional
    per‑symbol files ``calendar_<SYMBOL>.json``) into the MT5 *Common Files*
    directory.  This class abstracts the file handling, caching and basic
    filtering logic so the rest of the system can work with plain Python
    objects.
    """

    #: name of the main export file written by the bridge
    _CALENDAR_FILE = "calendar_data.json"

    #: per‑symbol file pattern (e.g. ``calendar_EURUSD.json``)
    _SYMBOL_FILE_FMT = "calendar_{symbol}.json"

    #: how long a cached read is considered fresh (seconds)
    _CACHE_TIMEOUT = 60

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialise the reader.

        Args:
            data_path: Absolute path to the MT5 *Common Files* directory.
                       If ``None`` the class attempts to locate it
                       automatically.
        """
        # Locate the MT5 common files directory if the caller did not
        # supply one.
        if data_path is None:
            data_path = self._find_mt5_common_path()

        self.data_path: Optional[Path] = Path(data_path) if data_path else None
        self._last_read: Optional[datetime] = None
        self._cached_data: Optional[Dict] = None

        logger.info("CalendarDataReader initialised")
        if self.data_path:
            logger.info(f"Data path resolved to: {self.data_path}")

    # ------------------------------------------------------------------
    # 1️⃣  Helpers to locate the MT5 *Common Files* folder
    # ------------------------------------------------------------------
    def _find_mt5_common_path(self) -> Optional[str]:
        """Try a handful of well‑known locations for the MT5 common files."""
        possible_paths = [
            os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files"),
            os.path.expandvars(
                r"C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
            ),
            "/home/user/.wine/drive_c/users/user/Application Data/MetaQuotes/Terminal/Common/Files",
        ]

        for path in possible_paths:
            if os.path.isdir(path):
                logger.info(f"Found MT5 common path: {path}")
                return path

        logger.warning("MT5 common files directory not found automatically")
        return None

    # ------------------------------------------------------------------
    # 2️⃣  Core file‑reading logic (with simple caching)
    # ------------------------------------------------------------------
    def _read_json_file(self, file_path: Path) -> Optional[Dict]:
        """Read a JSON file and return its content as a dict."""
        try:
            with open(file_path, "r", encoding="utf‑8") as f:
                return json.load(f)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Failed to read JSON file {file_path}: {exc}")
            return None

    def read_calendar_data(self, max_age_seconds: int = 300) -> Optional[Dict]:
        """
        Read the main calendar export.

        Args:
            max_age_seconds: Maximum age of the file to be considered fresh.
                             Stale files are ignored and ``None`` is returned.

        Returns:
            Parsed JSON dict or ``None`` if the file is missing / stale /
            unreadable.
        """
        # ------------------------------------------------------------------
        # 2.1  Return cached data if it is still fresh
        # ------------------------------------------------------------------
        if self._cached_data and self._last_read:
            age = (datetime.utcnow() - self._last_read).total_seconds()
            if age < self._CACHE_TIMEOUT:
                return self._cached_data

        # ------------------------------------------------------------------
        # 2.2  Validate that we know where the data lives
        # ------------------------------------------------------------------
        if not self.data_path:
            logger.warning("CalendarDataReader: no data_path configured")
            return None

        file_path = self.data_path / self._CALENDAR_FILE
        if not file_path.is_file():
            logger.warning(f"Calendar file not found: {file_path}")
            return None

        # ------------------------------------------------------------------
        # 2.3  Discard stale files (older than ``max_age_seconds``)
        # ------------------------------------------------------------------
        file_mtime = datetime.utcfromtimestamp(file_path.stat().st_mtime)
        age = (datetime.utcnow() - file_mtime).total_seconds()
        if age > max_age_seconds:
            logger.warning(
                f"Calendar data is stale ({int(age)} s old, "
                f"max allowed {max_age_seconds}s)"
            )
            return None

        # ------------------------------------------------------------------
        # 2.4  Load the JSON payload
        # ------------------------------------------------------------------
        data = self._read_json_file(file_path)
        if data is not None:
            self._cached_data = data
            self._last_read = datetime.utcnow()
        return data

    # ------------------------------------------------------------------
    # 3️⃣  Per‑symbol helper (optional – used by some strategies)
    # ------------------------------------------------------------------
    def read_symbol_check(self, symbol: str) -> Optional[Dict]:
        """
        Load a symbol‑specific JSON file (if the bridge creates one).

        Args:
            symbol: Trading symbol, e.g. ``EURUSD``

        Returns:
            Parsed JSON dict or ``None`` if the file does not exist.
        """
        if not self.data_path:
            return None

        filename = self._SYMBOL_FILE_FMT.format(symbol=symbol.upper())
        file_path = self.data_path / filename
        if not file_path.is_file():
            return None

        return self._read_json_file(file_path)

    # ------------------------------------------------------------------
    # 4️⃣  High‑level query helpers used by the trading engine
    # ------------------------------------------------------------------
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """
        Return a list of upcoming news events (within ``hours_ahead``).

        The bridge JSON uses the same field names as ``NewsEvent`` – we
        translate them into proper dataclass instances.
        """
        raw = self.read_calendar_data()
        if not raw or "events" not in raw:
            return []

        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        events: List[NewsEvent] = []

        # Mapping from the bridge's textual importance to the enum
        impact_map = {
            "High": NewsImpact.HIGH,
            "Medium": NewsImpact.MODERATE,
            "Low": NewsImpact.LOW,
            "None": NewsImpact.NONE,
        }

        for ev in raw["events"]:
            ev_time = datetime.utcfromtimestamp(ev.get("time", 0))
            if ev_time > cutoff:
                continue

            impact = impact_map.get(ev.get("importance", "None"), NewsImpact.NONE)

            events.append(
                NewsEvent(
                    time=ev_time,
                    currency=ev.get("currency", ""),
                    country=ev.get("country", ""),
                    event_name=ev.get("name", ""),
                    impact=impact,
                    sector=ev.get("sector", ""),
                    event_id=int(ev.get("event_id", 0)),
                    forecast=ev.get("forecast"),
                    previous=ev.get("previous"),
                    actual=ev.get("actual"),
                    minutes_until=int(ev.get("minutes_until", 0)),
                    is_active=bool(ev.get("is_active", False)),
                )
            )
        return events

    def should_avoid_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        Decide whether trading should be avoided for ``symbol`` based on the
        exported bridge data.

        Returns:
            (should_avoid, reason)
        """
        # 1️⃣  Symbol‑specific file takes precedence
        sym_data = self.read_symbol_check(symbol)
        if sym_data:
            return (
                bool(sym_data.get("should_avoid", False)),
                sym_data.get("reason", "No reason supplied"),
            )

        # 2️⃣  Fall back to the generic calendar data
        data = self.read_calendar_data()
        if not data:
            return False, "No calendar data available"

        # Extract the base and quote currencies from the symbol
        if len(symbol) < 6:
            return False, "Invalid symbol format"

        base_cur = symbol[:3].upper()
        quote_cur = symbol[3:6].upper()

        now = datetime.utcnow()
        for ev in data.get("events", []):
            # Skip inactive events
            if not ev.get("is_active"):
                continue

            ev_cur = ev.get("currency", "").upper()
            if ev_cur not in (base_cur, quote_cur):
                continue

            minutes = int(ev.get("minutes_until", 999))
            name = ev.get("name", "Unnamed")
            impact = ev.get("importance", "None")

            if minutes > 0:
                reason = f"{name} in {minutes} min [{ev_cur}] ({impact})"
            else:
                reason = f"{name} {abs(minutes)} min ago [{ev_cur}] ({impact})"

            return True, reason

        return False, "No major news scheduled"

    def get_risk_multiplier(self, symbol: str) -> float:
        """
        Return a risk multiplier (0.0 – 1.0) based on the bridge data.

        If the bridge says we should avoid trading we return ``0.0``;
        otherwise we return the explicit ``risk_multiplier`` field if it
        exists, falling back to ``1.0``.
        """
        sym_data = self.read_symbol_check(symbol)
        if sym_data:
            if sym_data.get("should_avoid"):
                return 0.0
            return float(sym_data.get("risk_multiplier", 1.0))

        # Generic fallback – if the generic calendar says “avoid”, block.
        avoid, _ = self.should_avoid_trading(symbol)
        return 0.0 if avoid else 1.0

    def is_bridge_active(self) -> bool:
        """
        Quick health‑check: the bridge is considered active if the main
        JSON file exists and has been modified within the last five minutes.
        """
        if not self.data_path:
            return False

        file_path = self.data_path / self._CALENDAR_FILE
        if not file_path.is_file():
            return False

        age = (datetime.utcnow() - datetime.utcfromtimestamp(file_path.stat().st_mtime)).total_seconds()
        return age < 300   # 5 minutes

# ----------------------------------------------------------------------
# 5️⃣  IntegratedCalendar – drops‑in replacement for MT5EconomicCalendar
# ----------------------------------------------------------------------
class IntegratedCalendar(MT5EconomicCalendar):
    """
    Calendar implementation that prefers the MT5 bridge data when it is
    available, otherwise falls back to the original ``MT5EconomicCalendar``
    logic (which queries MT5 directly).

    All public methods of ``MT5EconomicCalendar`` are preserved.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = CalendarDataReader()
        self.use_bridge = self.reader.is_bridge_active()

        if self.use_bridge:
            logger.info("✓ MT5 calendar bridge is ACTIVE – using real‑time data")
        else:
            logger.warning(
                "⚠ MT5 calendar bridge NOT active – falling back to MT5 API"
            )

    # ------------------------------------------------------------------
    # Overridden methods – they delegate to the bridge when possible
    # ------------------------------------------------------------------
    def should_avoid_trading(self, symbol: str) -> Tuple[bool, str, Optional[NewsEvent]]:
        """
        Return ``(should_avoid, reason, event)``.  When the bridge is
        active we use its data; otherwise we defer to the parent class.
        """
        if self.use_bridge:
            avoid, reason = self.reader.should_avoid_trading(symbol)
            # Try to fetch the concrete event (if any) for richer info
            upcoming = self.reader.get_upcoming_events(hours_ahead=2)
            active_event = next(
                (e for e in upcoming if e.is_active and e.currency in (symbol[:3], symbol[3:6])),
                None,
            )
            return avoid, reason, active_event

        # Fallback to the original implementation
        return super().should_avoid_trading(symbol)

    def get_risk_multiplier(self, symbol: str) -> float:
        """Return a multiplier (0‑1) – bridge takes precedence."""
        if self.use_bridge:
            return self.reader.get_risk_multiplier(symbol)
        return super().get_risk_multiplier(symbol)

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """Return upcoming events – bridge first, fallback second."""
        if self.use_bridge:
            return self.reader.get_upcoming_events(hours_ahead=hours_ahead)
        return super().get_upcoming_events(hours_ahead=hours_ahead)

# ----------------------------------------------------------------------
# 6️⃣  Small Flask API – useful for Grafana / external tooling
# ----------------------------------------------------------------------
class CalendarAPI:
    """
    Tiny Flask wrapper exposing the most useful endpoints:

    * ``GET /config`` – full JSON payload from the bridge
    * ``GET /events`` – upcoming events (query param ``hours``)
    * ``GET /avoid/<symbol>`` – should we avoid trading this symbol?
    * ``GET /multiplier/<symbol>`` – risk multiplier for the symbol
    * ``GET /healthz`` – simple health check
    """

    def __init__(self, reader: CalendarDataReader, host: str = "0.0.0.0", port: int = 5005):
        self.reader = reader
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        # ------------------------------------------------------------------
        # Routes
        # ------------------------------------------------------------------
        @self.app.route("/config", methods=["GET"])
        def config():
            data = self.reader.read_calendar_data()
            return jsonify(data or {})

        @self.app.route("/events", methods=["GET"])
        def events():
            hrs = int(request.args.get("hours", 24))
            evts = self.reader.get_upcoming_events(hours_ahead=hrs)
            # Serialize NewsEvent objects to plain dicts
            return jsonify([e.__dict__ for e in evts])

        @self.app.route("/avoid/<symbol>", methods=["GET"])
        def avoid(symbol: str):
            avoid, reason = self.reader.should_avoid_trading(symbol.upper())
            return jsonify({"avoid": avoid, "reason": reason})

        @self.app.route("/multiplier/<symbol>", methods=["GET"])
        def multiplier(symbol: str):
            mult = self.reader.get_risk_multiplier(symbol.upper())
            return jsonify({"multiplier": mult})

        @self.app.route("/healthz", methods=["GET"])
        def health():
            return "OK", 200

    def run(self):
        """Start the Flask development server (single‑threaded, suitable for internal use)."""
        logger.info(f"Starting Calendar API on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)


# ----------------------------------------------------------------------
# 7️⃣  Factory function – the rest of the codebase should call this
# ----------------------------------------------------------------------
def create_calendar(**kwargs) -> MT5EconomicCalendar:
    """
    Factory that returns an ``IntegratedCalendar`` instance.

    Example::

        from calendar_data_reader import create_calendar
        calendar = create_calendar(avoid_minutes_before=30, avoid_minutes_after=60)
    """
    return IntegratedCalendar(**kwargs)


# ----------------------------------------------------------------------
# 8️⃣  Simple command‑line test harness
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ------------------------------------------------------------------
    # Initialise the reader and display a quick status report

   # ------------------------------------------------------------------
    # Initialise the reader and display a quick status report
    # ------------------------------------------------------------------
    reader = CalendarDataReader()

    # Show whether the MT5 bridge is currently active
    if reader.is_bridge_active():
        logger.info("✅ MT5 calendar bridge is ACTIVE – real‑time data will be used")
    else:
        logger.warning(
            "⚠️ MT5 calendar bridge NOT active – falling back to MT5 API "
            "(ensure the MQL5 bridge EA is running in MT5)"
        )

    # ------------------------------------------------------------------
    # 1️⃣  Load the latest calendar payload (if any) and print a summary
    # ------------------------------------------------------------------
    calendar_payload = reader.read_calendar_data()
    if calendar_payload:
        logger.info("🗂️ Calendar payload loaded")
        logger.info(f"   Timestamp   : {calendar_payload.get('timestamp', 'N/A')}")
        logger.info(f"   Event count : {len(calendar_payload.get('events', []))}")
    else:
        logger.info("📂 No calendar payload found or it is stale")

    # ------------------------------------------------------------------
    # 2️⃣  Show upcoming high‑impact events for the next 24 h
    # ------------------------------------------------------------------
    upcoming = reader.get_upcoming_events(hours_ahead=24)
    logger.info("\n📅 Upcoming HIGH‑IMPACT events (next 24 h)")
    logger.info("-" * 60)
    if upcoming:
        for ev in upcoming[:10]:   # display at most 10 events
            mins = ev.minutes_until
            when = "now" if mins == 0 else f"in {mins} min"
            logger.info(
                f"[{ev.time.strftime('%Y-%m-%d %H:%M')}] "
                f"{ev.event_name} ({ev.currency}) – {when} – {ev.impact.name}"
            )
    else:
        logger.info("   No high‑impact events in the next 24 h")
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # 3️⃣  Symbol‑specific checks (example symbols)
    # ------------------------------------------------------------------
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    logger.info("\n🔎 Symbol‑specific news checks")
    logger.info("-" * 60)
    for sym in test_symbols:
        avoid, reason = reader.should_avoid_trading(sym)
        mult = reader.get_risk_multiplier(sym)
        logger.info(
            f"{sym:6} → avoid: {avoid:5} | multiplier: {mult:.2f} | reason: {reason}"
        )
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # 4️⃣  Optional: start the tiny Flask API so external tools
    #     (Grafana, alerts, etc.) can query the bridge data.
    #     The API is only started when the environment variable
    #     `CALENDAR_API_ENABLE` is set to a truthy value.
    # ------------------------------------------------------------------
    if os.getenv("CALENDAR_API_ENABLE", "0") in ("1", "true", "TRUE", "yes", "YES"):
        api = CalendarAPI(reader, host="0.0.0.0", port=5005)
        # Run the Flask server in a background thread so the script can
        # continue (or simply block here if you only want the API).
        from threading import Thread

        def _run_api():
            api.run()

        Thread(target=_run_api, daemon=True, name="calendar-api").start()
        logger.info("🚀 Calendar API is now listening on http://0.0.0.0:5005")
        # Keep the main thread alive so the API remains reachable.
        try:
            while True:
                # Simple heartbeat – could be replaced with a proper
                # async event loop if desired.
                import time

                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("🛑 Calendar API stopped by user")
    else:
        logger.info(
            "ℹ️ Calendar API not started – set CALENDAR_API_ENABLE=1 to enable it."
        )
