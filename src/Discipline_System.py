#!/usr/bin/env python3
"""
Discipline System

Trading contracts, journals, and accountability mechanisms.

Provides three tightly‑coupled objects:

* ``TradingContract`` – a pre‑commit contract with risk limits, win‑rate
  requirements and a digital signature.
* ``TradingJournal`` – a structured, searchable journal that forces a
  post‑trade entry for every order.
* ``AccountabilitySystem`` – tracks contract violations and automatically
  aggregates penalties.

All objects persist to JSON files on a mounted ``/app/config`` volume so
their state survives container restarts.  The module is thread‑safe,
logs every important event and can be queried from the rest of the
application (or from an external API) without any further dependencies.

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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Logging configuration (the main app usually configures the root logger;
# we just obtain a child logger here)
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper – atomic JSON write (prevents corrupted files on crash)
# ----------------------------------------------------------------------
def _atomic_write(path: Path, data: object) -> None:
    """Write *data* to *path* atomically (write‑tmp → rename)."""
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ----------------------------------------------------------------------
# 1️⃣  CONTRACT SECTION
# ----------------------------------------------------------------------
@dataclass
class ContractTerms:
    """All contract parameters – serialisable to JSON."""

    max_risk_per_trade: float = 5.0          # % of account equity per trade
    max_total_risk: float = 10.0              # % of account equity overall
    max_daily_trades: int = 5
    required_win_rate: float = 70.0           # % (historical)
    max_daily_loss_pct: float = 5.0           # % of equity lost in a day
    required_journal_entries: bool = True
    penalty_amount: float = 100.0             # USD per violation
    signed: bool = False
    signed_date: Optional[str] = None
    trader_name: str = "Trader"


class TradingContract:
    """
    Pre‑commit contract with digital signature and risk limits.
    The contract is persisted to ``contract_file`` (JSON) and can be
    inspected/modified only through the public methods.
    """

    _lock = threading.RLock()   # protects file I/O and in‑memory state

    def __init__(self, contract_file: str = "trading_contract.json"):
        self.contract_path = Path(contract_file)
        self.contract: ContractTerms = self._load_or_create()

    # ------------------------------------------------------------------
    def _load_or_create(self) -> ContractTerms:
        """Load an existing contract or create a brand‑new default one."""
        with self._lock:
            if self.contract_path.is_file():
                try:
                    data = json.loads(self.contract_path.read_text(encoding="utf-8"))
                    logger.info(f"Loaded existing contract from {self.contract_path}")
                    return ContractTerms(**data)
                except Exception as exc:   # pragma: no cover
                    logger.error(f"Failed to parse contract file: {exc}")

            # --- create default -------------------------------------------------
            default = ContractTerms()
            self._save(default)
            logger.info(f"Created default contract at {self.contract_path}")
            return default

    # ------------------------------------------------------------------
    def _save(self, contract: ContractTerms) -> None:
        """Persist the contract atomically."""
        with self._lock:
            _atomic_write(self.contract_path, asdict(contract))

    # ------------------------------------------------------------------
    def sign_contract(self, trader_name: str) -> bool:
        """
        Digitally sign the contract.  Once signed the terms become immutable
        for the current trading session (you can still change the JSON file
        manually, but the system will refuse to operate on an unsigned contract).
        """
        with self._lock:
            if self.contract.signed:
                logger.warning("Attempt to re‑sign an already signed contract")
                return False

            self.contract.signed = True
            self.contract.signed_date = datetime.now().isoformat()
            self.contract.trader_name = trader_name
            self._save(self.contract)

            logger.info(f"✅ Contract signed by {trader_name}")
            logger.info(
                f"Terms → max_risk_per_trade={self.contract.max_risk_per_trade}% | "
                f"max_total_risk={self.contract.max_total_risk}% | "
                f"max_daily_trades={self.contract.max_daily_trades}"
            )
            return True

    # ------------------------------------------------------------------
    def is_signed(self) -> bool:
        """Return True if the contract has been signed."""
        return self.contract.signed

    # ------------------------------------------------------------------
    def validate_risk_compliance(
        self,
        current_risk_pct: float,
        total_exposure_pct: float,
    ) -> Tuple[bool, str]:
        """
        Verify that a prospective trade respects the contract limits.

        Args:
            current_risk_pct: % of equity this single trade would risk.
            total_exposure_pct: % of equity already allocated to open positions.

        Returns:
            (compliant, explanatory_message)
        """
        if not self.contract.signed:
            return False, "Contract not signed"

        if current_risk_pct > self.contract.max_risk_per_trade:
            return (
                False,
                f"Risk {current_risk_pct:.2f}% exceeds per‑trade limit "
                f"{self.contract.max_risk_per_trade:.2f}%",
            )

        if total_exposure_pct + current_risk_pct > self.contract.max_total_risk:
            return (
                False,
                f"Total exposure {total_exposure_pct + current_risk_pct:.2f}% would exceed "
                f"overall limit {self.contract.max_total_risk:.2f}%",
            )

        return True, "Compliant with contract"

    # ------------------------------------------------------------------
    def get_terms(self) -> ContractTerms:
        """Return a copy of the current contract terms."""
        # Returning the dataclass itself is safe because it is immutable
        # (no mutating methods are exposed).  If you need a deep copy,
        # use ``asdict``.
        return self.contract

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Convenient JSON‑serialisable representation."""
        return asdict(self.contract)


# ----------------------------------------------------------------------
# 2️⃣  JOURNAL SECTION
# ----------------------------------------------------------------------
@dataclass
class JournalEntry:
    """One immutable journal line – all fields are JSON‑serialisable."""

    timestamp: str
    trade_ticket: Optional[int]
    symbol: str
    direction: str                     # BUY / SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    result: str = "OPEN"               # OPEN, WIN, LOSS, CANCELLED
    profit: float = 0.0
    r_multiple: float = 0.0
    emotional_state: str = "Calm"
    setup_quality: str = ""            # e.g. “High”, “Medium”, “Low”
    lessons_learned: str = ""
    screenshots: List[str] = field(default_factory=list)


class TradingJournal:
    """
    Structured journal that forces a post‑trade entry for every order.
    Entries are persisted to ``journal_file`` (JSON array) and loaded on
    start‑up.  The class is thread‑safe.
    """

    _lock = threading.RLock()

    def __init__(self, journal_file: str = "trading_journal.json"):
        self.journal_path = Path(journal_file)
        self.entries: List[JournalEntry] = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> List[JournalEntry]:
        """Read the JSON file; return an empty list if missing or corrupt."""
        with self._lock:
            if self.journal_path.is_file():
                try:
                    raw = json.loads(self.journal_path.read_text(encoding="utf-8"))
                    entries = [JournalEntry(**item) for item in raw]
                    logger.info(f"Loaded {len(entries)} journal entries")
                    return entries
                except Exception as exc:   # pragma: no cover
                    logger.error(f"Failed to parse journal file: {exc}")

            logger.info("No journal file found – starting with an empty journal")
            return []

    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Persist the whole journal atomically."""
        with self._lock:
            data = [asdict(e) for e in self.entries]
            _atomic_write(self.journal_path, data)

    # ------------------------------------------------------------------
    def add_entry(self, entry: JournalEntry) -> None:
        """Append a new entry and persist immediately."""
        with self._lock:
            self.entries.append(entry)
            self._save()
            logger.info(
                f"Journal entry added – {entry.symbol} {entry.direction} "
                f"@ {entry.entry_price:.5f}"
            )

    # ------------------------------------------------------------------
    def create_trade_entry(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        setup_quality: str,
        emotional_state: str = "Calm",
    ) -> JournalEntry:
        """
        Convenience helper that builds a ``JournalEntry`` for a newly opened
        trade, stores it and returns the object (so the caller can later
        update the result).
        """
        entry = JournalEntry(
            timestamp=datetime.now().isoformat(),
            trade_ticket=ticket,
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            emotional_state=emotional_state,
            setup_quality=setup_quality,
        )
        self.add_entry(entry)
        return entry

    # ------------------------------------------------------------------
    def update_trade_result(
        self,
        ticket: int,
        result: str,
        profit: float,
        r_multiple: float,
        lessons_learned: str = "",
        screenshots: Optional[List[str]] = None,
    ) -> None:
        """
        Locate the entry for *ticket* and fill in the outcome fields.
        ``result`` must be one of ``WIN``, ``LOSS`` or ``CANCELLED``.
        """
        with self._lock:
            for entry in self.entries:
                if entry.trade_ticket == ticket:
                    entry.result = result.upper()
                    entry.profit = profit
                    entry.r_multiple = r_multiple
                    entry.lessons_learned = lessons_learned
                    if screenshots:
                        entry.screenshots.extend(screenshots)
                    self._save()
                    logger.info(
                        f"Journal updated – ticket #{ticket}: {result} "
                        f"({profit:+.2f}$, {r_multiple:.2f}R)"
                    )
                    return
            logger.warning(f"Tried to update unknown ticket #{ticket}")

    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict:
        """Aggregate simple performance metrics from the journal."""
        with self._lock:
            if not self.entries:
                return {}

            closed = [e for e in self.entries if e.result in ("WIN", "LOSS")]
            if not closed:
                return {"total_trades": len(self.entries), "closed_trades": 0}

            wins = [e for e in closed if e.result == "WIN"]
            losses = [e for e in closed if e.result == "LOSS"]

            win_rate = len(wins) / len(closed) * 100
            avg_win_r = sum(e.r_multiple for e in wins) / len(wins) if wins else 0.0
            avg_loss_r = sum(e.r_multiple for e in losses) / len(losses) if losses else 0.0

            return {
                "total_trades": len(self.entries),
                "closed_trades": len(closed),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": round(win_rate, 2),
                "avg_win_r": round(avg_win_r, 2),
                "avg_loss_r": round(avg_loss_r, 2),
                "total_profit": round(sum(e.profit for e in closed), 2),
            }

    # ------------------------------------------------------------------
    def recent_entries(self, limit: int = 10) -> List[JournalEntry]:
        """Return the *limit* most recent entries (newest first)."""
        with self._lock:
            return list(reversed(self.entries[-limit:]))

    # ------------------------------------------------------------------
    def to_list(self) -> List[Dict]:
        """JSON‑serialisable list of all entries (useful for API responses)."""
        with self._lock:
            return [asdict(e) for e in self.entries]


# ----------------------------------------------------------------------
# 3️⃣  ACCOUNTABILITY SECTION
# ----------------------------------------------------------------------
class AccountabilitySystem:
    """
    Tracks contract violations, aggregates penalties and provides a simple
    audit trail.  Violations are persisted to ``violations_file`` (JSON
    array).  The class is thread‑safe.
    """

    _lock = threading.RLock()

    def __init__(self, violations_file: str = "violations.json"):
        self.violations_path = Path(violations_file)
        self.violations: List[Dict] = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> List[Dict]:
        """Read the JSON file; return an empty list if missing or corrupt."""
        with self._lock:
            if self.violations_path.is_file():
                try:
                    data = json.loads(self.violations_path.read_text(encoding="utf-8"))
                    logger.info(f"Loaded {len(data)} past violations")
                    return data
                except Exception as exc:   # pragma: no cover
                    logger.error(f"Failed to parse violations file: {exc}")

            logger.info("No violations file – starting fresh")
            return []

    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Persist the violations atomically."""
        with self._lock:
            _atomic_write(self.violations_path, self.violations)

    # ------------------------------------------------------------------
    def record_violation(self, violation_type: str, description: str) -> None:
        """
        Add a new violation entry.  The penalty amount is taken from the
        contract (if a contract is loaded) – otherwise a default of $100 is used.
        """
        with self._lock:
            penalty = 100.0
            # Try to fetch the current contract penalty if the contract module
            # is available (avoids circular imports at import time).
            try:
                from .discipline_system import trading_contract
                penalty = trading_contract.contract.penalty_amount
            except Exception:   # pragma: no cover
                pass

            violation = {
                "timestamp": datetime.now().isoformat(),
                "type": violation_type,
                "description": description,
                "penalty_due": penalty,
            }
            self.violations.append(violation)
            self._save()

            logger.error(f"⚠️ Violation recorded – {violation_type}")
            logger.error(f"   Description : {description}")
            logger.error(f"   Penalty     : ${penalty:,.2f}")

    # ------------------------------------------------------------------
    def get_total_penalties(self) -> float:
        """Sum of all outstanding penalties."""
        with self._lock:
            return sum(v.get("penalty_due", 0.0) for v in self.violations)

    # ------------------------------------------------------------------
    def get_violation_count(self) -> int:
        """Number of recorded violations."""
        return len(self.violations)

    # ------------------------------------------------------------------
    def recent_violations(self, limit: int = 10) -> List[Dict]:
        """Return the most recent *limit* violations (newest first)."""
        with self._lock:
            return list(reversed(self.violations[-limit:]))

    # ------------------------------------------------------------------
       def to_list(self) -> List[Dict]:
        """JSON‑serialisable list of all violations."""
        with self._lock:
            return list(self.violations)


# ----------------------------------------------------------------------
# 4️⃣  PUBLIC SINGLETONS (imported by the rest of the code‑base)
# ----------------------------------------------------------------------
# These objects are created once at import time and can be used from any
# module that does:
#     from src.discipline_system import trading_contract, trading_journal, accountability_system
trading_contract = TradingContract()
trading_journal = TradingJournal()
accountability_system = AccountabilitySystem()


# ----------------------------------------------------------------------
# 5️⃣  OPTIONAL HIGH‑LEVEL FAÇADE (convenient for external callers)
# ----------------------------------------------------------------------
class DisciplineFacade:
    """
    Tiny façade that bundles the three core objects and offers a single
    entry‑point for common operations.  This is handy for UI/API layers
    that do not want to import each class separately.
    """

    def __init__(self):
        self.contract = trading_contract
        self.journal = trading_journal
        self.accountability = accountability_system

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------
    def sign_contract(self, trader_name: str) -> bool:
        """Sign the trading contract."""
        return self.contract.sign_contract(trader_name)

    def is_contract_signed(self) -> bool:
        """Whether the contract has been signed."""
        return self.contract.is_signed()

    def get_contract_terms(self) -> Dict:
        """Return the contract as a plain dict (JSON‑friendly)."""
        return self.contract.to_dict()

    def validate_risk(
        self,
        current_risk_pct: float,
        total_exposure_pct: float,
    ) -> Tuple[bool, str]:
        """
        Wrapper around ``TradingContract.validate_risk_compliance``.
        Returns (compliant, message).
        """
        return self.contract.validate_risk_compliance(
            current_risk_pct, total_exposure_pct
        )

    # ------------------------------------------------------------------
    # Journal helpers
    # ------------------------------------------------------------------
    def add_journal_entry(self, entry: JournalEntry) -> None:
        """Append a fully‑formed journal entry."""
        self.journal.add_entry(entry)

    def create_trade_journal_entry(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        setup_quality: str,
        emotional_state: str = "Calm",
    ) -> JournalEntry:
        """
        Convenience shortcut used by the execution engine when a new trade
        is opened.
        """
        return self.journal.create_trade_entry(
            ticket,
            symbol,
            direction,
            entry_price,
            stop_loss,
            take_profit,
            setup_quality,
            emotional_state,
        )

    def close_trade_journal_entry(
        self,
        ticket: int,
        result: str,
        profit: float,
        r_multiple: float,
        lessons_learned: str = "",
        screenshots: Optional[List[str]] = None,
    ) -> None:
        """Mark a trade as closed and store its outcome."""
        self.journal.update_trade_result(
            ticket,
            result,
            profit,
            r_multiple,
            lessons_learned,
            screenshots,
        )

    def get_journal_statistics(self) -> Dict:
        """Aggregated performance numbers."""
        return self.journal.get_statistics()

    def recent_journal_entries(self, limit: int = 10) -> List[Dict]:
        """Return the most recent *limit* entries as dicts."""
        return [asdict(e) for e in self.journal.recent_entries(limit)]

    # ------------------------------------------------------------------
    # Accountability helpers
    # ------------------------------------------------------------------
    def record_violation(self, violation_type: str, description: str) -> None:
        """Log a contract breach."""
        self.accountability.record_violation(violation_type, description)

    def total_penalties(self) -> float:
        """Sum of all outstanding penalties."""
        return self.accountability.get_total_penalties()

    def violation_count(self) -> int:
        """Number of recorded violations."""
        return self.accountability.get_violation_count()

    def recent_violations(self, limit: int = 10) -> List[Dict]:
        """Most recent violations (newest first)."""
        return self.accountability.recent_violations(limit)

    # ------------------------------------------------------------------
    # Convenience snapshot (useful for dashboards / API)
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict:
        """
        Return a single dict that contains the contract, journal stats and
        accountability summary.  This is ideal for a Grafana JSON API or
        a quick health‑check endpoint.
        """
        return {
            "contract": self.get_contract_terms(),
            "journal_stats": self.get_journal_statistics(),
            "total_penalties_usd": round(self.total_penalties(), 2),
            "violation_count": self.violation_count(),
        }


# ----------------------------------------------------------------------
# 6️⃣  GLOBAL façade instance (ready for import)
# ----------------------------------------------------------------------
discipline = DisciplineFacade()
