# =============================================================================
# advanced_execution_engine.py
# =============================================================================
"""
AdvancedExecutionEngine – the heart of CQT.

It glues together:
* Risk‑management layer
* Multi‑venue broker adapters (MT5 / IBKR …)
* Guard classes (sentiment, calendar, volatility, shock)
* Regime classifier (HMM)
* Prometheus metrics (latency, slippage, depth‑guard outcomes)
* Shadow mode (records what would have happened without sending an order)

All heavyweight objects are instantiated once at process start so that the
async event‑loop can reuse them without recreating connections.
"""

# -------------------------------------------------------------------------
# Standard library
# -------------------------------------------------------------------------
import asyncio
import importlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytz

# -------------------------------------------------------------------------
# Third‑party
# -------------------------------------------------------------------------
from prometheus_client import Counter, Gauge, Histogram

# -------------------------------------------------------------------------
# Local imports (relative to the package root)
# -------------------------------------------------------------------------
from src.config import Config, env, logger as cfg_logger
from src.risk_management.risk_manager import RiskManagementLayer
from src.market_data_manager import MarketDataManager
from src.bot_control import BotControl
from src.hmm_regime import HMMRegime
from src.guards import (
    SentimentGuard,
    CalendarLockout,
    VolatilityGuard,
    ShockDetector,
)
from src.guard_helpers import (
    check_depth,
    check_latency,
    check_spread,
    check_volatility,
)
from src.trade_logger import TradeLogger
from src.metrics.prometheus import (
    order_latency_seconds,
    order_price_slippage,
    depth_ok_gauge,
    depth_fail_gauge,
    trade_blocked_reason_total,
)
from src.utils.db import get_db_session
from src.utils.common import utc_now

# -------------------------------------------------------------------------
# Global logger – same logger is used throughout the process
# -------------------------------------------------------------------------
log = logging.getLogger(__name__)

# =============================================================================
# Helper – broker adapter factory
# =============================================================================
def load_adapter(broker_type: str, cfg: Dict[str, Any]) -> Any:
    """
    Dynamically import the broker adapter class and instantiate it.

    Supported types: mt5, ibkr, ctrader, ninjatrader, tradovate.
    """
    mapping = {
        "mt5": "mt5_adapter.MT5Adapter",
        "ibkr": "ibkr_adapter.IBKRAdapter",
        "ctrader": "ctrader_adapter.CTraderAdapter",
        "ninjatrader": "ninjatrader_adapter.NinjaTraderAdapter",
        "tradovate": "tradovate_adapter.TradovateAdapter",
    }

    if broker_type not in mapping:
        raise ValueError(f"Unsupported broker_type: {broker_type}")

    module_name, class_name = mapping[broker_type].split(".")
    mod = importlib.import_module(f"src.{module_name}")
    klass = getattr(mod, class_name)
    return klass(cfg)


# =============================================================================
# AdvancedExecutionEngine
# =============================================================================
class AdvancedExecutionEngine:
    """
    Core execution engine that:

    1️⃣ Evaluates all pre‑trade guards.  
    2️⃣ Consults the HMM regime classifier.  
    3️⃣ Computes lot size from the risk manager.  
    4️⃣ Selects the appropriate broker (MT5 / IBKR …).  
    5️⃣ Sends the order (or records a shadow entry).  
    6️⃣ Updates Prometheus metrics.  
    7️⃣ Logs everything via the shared TradeLogger.
    """

    # -----------------------------------------------------------------
    # Constructor – heavy objects are created once
    # -----------------------------------------------------------------
    def __init__(self, *, shadow: bool = False, **_unused_kwargs: Any):
        # -----------------------------------------------------------------
        # 0️⃣ Core configuration & helpers
        # -----------------------------------------------------------------
        self.cfg = Config().settings
        self.shadow = shadow
        self.tz = pytz.UTC

        # -----------------------------------------------------------------
        # 1️⃣ Initialise the risk‑management layer (needs a DB session)
        # -----------------------------------------------------------------
        db_session = get_db_session()
        self.risk_mgr = RiskManagementLayer(db_session)

        # -----------------------------------------------------------------
        # 2️⃣ Bot control (pause / resume / kill‑switch)
        # -----------------------------------------------------------------
        self.bot_ctrl = BotControl()

        # -----------------------------------------------------------------
        # 3️⃣ Market‑data manager (provides depth, recent ticks, etc.)
        # -----------------------------------------------------------------
        self.market = MarketDataManager(self.cfg)

        # -----------------------------------------------------------------
        # 4️⃣ Regime classifier (HMM) – may fall back to a dummy classifier
        # -----------------------------------------------------------------
        self.regime_clf = HMMRegime(self.cfg)
        if not self.regime_clf.load():
            cfg_logger.warning(
                "HMM model not found – falling back to dummy regime classifier"
            )
            self.regime_clf = None  # treat as always "neutral"

        # -----------------------------------------------------------------
        # 5️⃣ Guard instances (sentiment, calendar, volatility, shock)
        # -----------------------------------------------------------------
        self.sentiment_guard = SentimentGuard(self.cfg, redis_client=None)
        self.calendar_guard = CalendarLockout(self.cfg, calendar_module=None)
        self.vol_guard = VolatilityGuard(self.cfg)
        self.shock_guard = ShockDetector(self.cfg)

        # -----------------------------------------------------------------
        # 6️⃣ Broker adapters – primary & secondary (fallback)
        # -----------------------------------------------------------------
        primary_type = self.cfg.get("broker_primary", "mt5").lower()
        secondary_type = self.cfg.get("broker_secondary", "ibkr").lower()
        self.primary_broker = load_adapter(primary_type, self.cfg)
        self.secondary_broker = load_adapter(secondary_type, self.cfg)

        # -----------------------------------------------------------------
        # 7️⃣ Prometheus counters for shadow mode
        # -----------------------------------------------------------------
        self.shadow_counter = Counter(
            "cqt_shadow_orders_total",
            "Orders that would have been sent in shadow mode",
            ["symbol", "direction"],
        )

        # -----------------------------------------------------------------
        # 8️⃣ Trade logger (singleton – writes to SQLite + optional JSON)
        # -----------------------------------------------------------------
        self.trade_logger = TradeLogger()  # already a singleton

        # -----------------------------------------------------------------
        # 9️⃣ Miscellaneous internal state
        # -----------------------------------------------------------------
        self._pending_submissions: Dict[str, Dict[str, Any]] = {}

        cfg_logger.info(
            f"AdvancedExecutionEngine initialised (shadow={self.shadow})"
        )

    # -----------------------------------------------------------------
    # Helper – compute lot size from a USD stake
    # -----------------------------------------------------------------
    @staticmethod
    def _lot_from_stake(stake_usd: float, contract_notional: float = 100_000) -> float:
        """
        Convert a dollar stake into a lot size.
        By default we assume 1 lot = 100 000 units (standard FX lot).
        """
        return stake_usd / contract_notional

    # -----------------------------------------------------------------
    # Helper – choose broker based on symbol
    # -----------------------------------------------------------------
    def _choose_broker(self, symbol: str) -> Any:
        """
        Simple routing rule:
        * Symbols ending with "USD" → primary broker (MT5 by default)
        * Everything else → secondary broker (IBKR by default)
        """
        return (
            self.primary_broker
            if symbol.upper().endswith("USD")
            else self.secondary_broker
        )

    # -----------------------------------------------------------------
    # Helper – regime multiplier (risk scaling)
    # -----------------------------------------------------------------
    def _regime_multiplier(self, bar: Dict[str, Any]) -> float:
        """
        Returns a risk multiplier based on the current regime:
        * high‑vol regime → 0.5 (more conservative)
        * medium → 1.0
        * low → 1.5 (more aggressive)

        If the HMM model is unavailable we fall back to 1.0.
        """
        if not self.regime_clf:
            return 1.0

        regime = self.regime_clf.predict(bar)
        if regime == 2:      # high volatility
            return 0.5
        elif regime == 1:    # medium
            return 1.0
        else:                # low volatility
            return 1.5

    # -----------------------------------------------------------------
    # Helper – human‑readable regime label (for logging / metrics)
    # -----------------------------------------------------------------
    def _current_regime_label(self) -> str:
        """
        Returns a string label for the regime that the HMM classifier
        currently predicts. If the model is unavailable we return
        ``"unknown"``.
        """
        if not self.regime_clf:
            return "unknown"
        latest_bar = self.market.latest_bar()
        regime_id = self.regime_clf.predict(latest_bar)
        mapping = {0: "low", 1: "medium", 2: "high"}
        return mapping.get(regime_id, "unknown")

    # -----------------------------------------------------------------
    # Pre‑trade guard – runs all safety checks synchronously
    # -----------------------------------------------------------------
    def _pre_trade_guard(self, signal: Dict[str, Any]) -> bool:
        """
        Returns True iff *all* guards approve the trade.

        Expected keys in ``signal``:
            - ``symbol``
            - ``volume`` (positive for BUY, negative for SELL)
            - ``required_volume`` (USD amount you intend to risk)
        """
        symbol = signal["symbol"]
        volume = signal["volume"]
        required_usd = signal["required_volume"]

        # ---- Depth / LIR guard ----
        if not check_depth(
            broker=self.primary_broker,
            symbol=symbol,
            required_volume=abs(volume),
            min_lir=self.cfg.get("risk", {}).get("lir_min_abs", 0.30),
        ):
            log.warning("Depth/LIR guard rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="depth_lir").inc()
            return False

        # ---- Latency guard ----
        if not check_latency(
            broker=self.primary_broker,
            max_latency_sec=self.cfg.get("risk", {}).get("max_latency_sec", 0.15),
        ):
            log.warning("Latency guard rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="latency").inc()
            return False

        # ---- Spread guard ----
        if not check_spread(
            broker=self.primary_broker,
            symbol=symbol,
            max_spread_pips=self.cfg.get("risk", {}).get("max_spread_pips", 0.5),
        ):
            log.warning("Spread guard rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="spread").inc()
            return False

        # ---- Volatility guard (ATR‑based) ----
        if not check_volatility(
            tech_calc=self.market,
            symbol=symbol,
            atr_multiplier=self.cfg.get("risk", {}).get("atr_multiplier", 2.0),
            max_atr_pct=self.cfg.get("risk", {}).get("max_atr_pct", 0.20),
        ):
            log.warning("Volatility guard rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="volatility").inc()
            return False

        # ---- Sentiment guard (optional) ----
        if not self.sentiment_guard.check(symbol):
            log.warning("Sentiment guard rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="sentiment").inc()
            return False

        # ---- Calendar lock‑out guard ----
        if self.calendar_guard.is_locked():
            log.warning("Calendar lock‑out active – trade skipped")
            trade_blocked_reason_total.labels(reason="calendar").inc()
            return False

        # ---- Volatility‑spike guard (custom) ----
        if not self.vol_guard.check(symbol):
            log.warning("Volatility‑spike guard rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="vol_spike").inc()
            return False

        # ---- Shock detector (spread, desync, liquidity) ----
        if not self.shock_guard.check(self.market.get_latest_snapshot(symbol)):
            log.warning("Shock detector rejected trade for %s", symbol)
            trade_blocked_reason_total.labels(reason="shock").inc()
            return False

        # All checks passed
        return True

    # -----------------------------------------------------------------
    # Core order‑submission entry point (sync – called from the signal loop)
    # -----------------------------------------------------------------
    def send_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected keys in ``signal``:
            - ``symbol``
            - ``side`` (``"BUY"`` / ``"SELL"``)
            - ``price`` (desired entry price)
            - ``volume`` (positive for BUY, negative for SELL)
            - ``required_volume`` (USD risk amount)
            - ``bar`` (latest OHLCV bar – needed for regime)
            - ``bucket_id`` (risk bucket identifier)
        Returns a dict compatible with the rest of the CQT codebase.
        """
        start_ts = time.time()
        symbol = signal["symbol"]
        side = signal["side"]
        price = signal.get("price")
        volume = signal["volume"]
        required_usd = signal["required_volume"]
        bar = signal.get("bar", {})
        bucket_id = signal.get("bucket_id")

        # -----------------------------------------------------------------
        # a) Bot state (pause / kill‑switch)
        # -----------------------------------------------------------------
        if self.bot_ctrl.is_paused or self.bot_ctrl.kill_switch_active:
            log.info("Bot paused or kill‑switch active – ignoring signal")
            return {"executed": False, "reason": "bot_paused_or_killed"}

        # -----------------------------------------------------------------
        # b) Run all pre‑trade guards
        # -----------------------------------------------------------------
        if not self._pre_trade_guard(signal):
            # Guard already logged the reason and incremented the counter.
            return {"executed": False, "reason": "guard_rejection"}

        # -----------------------------------------------------------------
        # c) Regime‑based risk multiplier
        # -----------------------------------------------------------------
        risk_multiplier = self._regime_multiplier(bar)

        # -----------------------------------------------------------------
        # d) Compute stake (USD) and translate to lot size
        # -----------------------------------------------------------------
        stake_usd = self.risk_mgr.compute_stake(
            bucket_id=signal["bucket_id"],
            equity=self.risk_mgr.current_equity(),
        )
        # Apply regime multiplier
        stake_usd *= risk_multiplier

        lot = self._lot_from_stake(
            stake_usd, contract_notional=self.cfg.get("contract_notional", 100_000)
        )
        # Enforce minimum lot size from config
        min_lot = self.cfg.get("min_lot", 0.01)
        lot = max(lot, min_lot)

        # -----------------------------------------------------------------
        # e) Choose broker (primary / secondary) and send the order
        # -----------------------------------------------------------------
        broker = self._choose_broker(symbol)

        # -----------------------------------------------------------------
        # f) Shadow mode handling
        # -----------------------------------------------------------------
        if self.shadow:
            # Simulate a fill with a tiny random jitter (±0.5 pip)
            jitter = random.uniform(-0.0005, 0.0005)
            fill_price = price + jitter if price is not None else None

            latency = time.time() - start_ts
            order_latency_seconds.labels(symbol=symbol, shadow="yes").observe(latency)

            # Slippage in pips
            if fill_price is not None:
                pip_factor = 0.0001
                if symbol.upper().endswith("JPY"):
                    pip_factor = 0.01
                slippage = abs(fill_price - price) / pip_factor
                order_price_slippage.labels(symbol=symbol, side=side).inc(slippage)
            else:
                slippage = None

            # Persist a JSON line to the shadow log
            shadow_entry = {
                "timestamp": utc_now().isoformat(),
                "symbol": symbol,
                "side": side,
                "requested_price": price,
                "fill_price": round(fill_price, 5) if fill_price else None,
                "lot": lot,
                "latency_seconds": round(latency, 4),
                "slippage_pips": round(slippage, 3) if slippage else None,
                "reason": "shadow_mode",
            }
            shadow_path = Path("/var/log/cqt/shadow.log")
            shadow_path.parent.mkdir(parents=True, exist_ok=True)
            with shadow_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(shadow_entry) + "\n")

            self.shadow_counter.labels(symbol=symbol, direction=side).inc()
            log.debug("[SHADOW] %s", shadow_entry)

            return {
                "executed": True,
                "shadow": True,
                "fill_price": fill_price,
                "lot": lot,
                "latency_seconds": latency,
                "slippage_pips": slippage,
            }

        # -----------------------------------------------------------------
        # g) REAL mode – actually send the order to the broker
        # -----------------------------------------------------------------
        try:
            broker_response = broker.send_order(
                symbol=symbol,
                side=side,
                price=price,
                volume=lot,
                stop_loss=signal.get("stop_loss"),
                take_profit=signal.get("take_profit"),
                comment="Citadel‑QT",
            )
        except Exception as exc:  # pragma: no cover – exercised via unit tests
            log.exception(
                "Broker %s raised exception for %s",
                broker.__class__.__name__,
                symbol,
            )
            # Record failure latency (still counts as a latency measurement)
            order_latency_seconds.labels(symbol=symbol, shadow="no").observe(
                time.time() - start_ts
            )
            return {"executed": False, "reason": f"broker_error: {exc}"}

        # -----------------------------------------------------------------
        # h) Record latency & slippage (if fill info is available)
        # -----------------------------------------------------------------
        latency = time.time() - start_ts
        order_latency_seconds.labels(symbol=symbol, shadow="no").observe(latency)

        fill_price = broker_response.get("fill_price")
        if fill_price is not None and price is not None:
            pip_factor = 0.0001
            if symbol.upper().endswith("JPY"):
                pip_factor = 0.01
            slippage = abs(fill_price - price) / pip_factor
            order_price_slippage.labels(symbol=symbol, side=side).inc(slippage)
        else:
            slippage = None

        # -----------------------------------------------------------------
        # i) Log the trade opening in the immutable ledger
        # -----------------------------------------------------------------
        self.trade_logger.log_trade_open(
            symbol=symbol,
            direction=side,
            entry_price=price,
            lot_size=lot,
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            entry_quality=int(risk_multiplier * 100),  # simple quality proxy
            confluence_score=signal.get("confluence_score", 0),
            mtf_alignment=signal.get("mtf_alignment", 0.0),
            market_regime=self._current_regime_label(),
            session=self.market.current_session(),
            volatility_state=self.market.current_vol_state(),
            rr_ratio=self.cfg.get("risk", {}).get("RR_target", 5.0),
            stack_level=signal.get("stack_level", 1),
            platform="MT5" if broker is self.primary_broker else "IBKR",
            magic_number=signal.get("magic_number", 0),
            comment=signal.get("comment", ""),
            metadata=json.dumps(
                {
                    "risk_multiplier": risk_multiplier,
                    "regime": self._current_regime_label(),
                    "guard_passed": True,
                }
            ),
        )

        # -----------------------------------------------------------------
        # j) Update the risk manager – deduct the risk‑fraction for this trade
        # -----------------------------------------------------------------
        if bucket_id is not None:
            # The risk manager will later adjust the bucket's remaining
            # risk‑fraction based on the eventual P&L.
            self.risk_mgr.record_successful_trade(bucket_id, required_usd)

        # -----------------------------------------------------------------
        # k) Emit Prometheus metrics for the successful order
        # -----------------------------------------------------------------
        depth_ok_gauge.labels(bucket_id=str(bucket_id), symbol=symbol).inc()
        # (If you want a separate failure counter, increment it in the
        #  exception block above.)

        # -----------------------------------------------------------------
        # l) Return a normalized response to the caller
        # -----------------------------------------------------------------
        result = {
            "executed": True,
            "shadow": False,
            "order_id": broker_response.get("order_id"),
            "fill_price": fill_price,
            "lot": lot,
            "latency_seconds": latency,
            "slippage_pips": slippage,
            "broker": broker.__class__.__name__,
            "risk_multiplier": risk_multiplier,
        }

        log.info(
            "[EXEC] Order sent – %s %s %.4f @ %.5f (lot=%.4f, latency=%.3fs, slippage=%s)",
            side,
            symbol,
            lot,
            price,
            lot,
            latency,
            f"{slippage:.2f} pips" if slippage is not None else "N/A",
        )
        return result

    # -----------------------------------------------------------------
    # Public API – entry point used by the signal engine
    # -----------------------------------------------------------------
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper that the higher‑level signal dispatcher calls.
        It performs a quick pause/kill‑switch check, runs the guard chain,
        and finally forwards the signal to ``send_order``.
        
        ✅ FIXED: Removed `async` keyword (was unused – no await operations)
        """
        # 0️⃣ Global bot state (pause / kill‑switch)
        if self.bot_ctrl.is_paused or self.bot_ctrl.kill_switch_active:
            log.info("Bot paused or kill‑switch active – signal ignored")
            return {"executed": False, "reason": "bot_paused_or_killed"}

        # 1️⃣ Run the full guard suite (synchronous)
        if not self._pre_trade_guard(signal):
            # Guard already logged why it was rejected.
            return {"executed": False, "reason": "guard_rejection"}

        # 2️⃣ All clear – forward to the order‑submission routine.
        return self.send_order(signal)
