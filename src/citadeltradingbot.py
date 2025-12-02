#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
import threading
import time
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import threading, time, os, signal

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5

# ----------------------------------------------------------------------
# Local imports (these modules already exist in the repo)
# ----------------------------------------------------------------------
from advanced_execution import AdvancedExecutionEngine
from risk_management import RiskManagementLayer
from dashboard_connector import DashboardConnector

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/citadel_trading_bot.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "citadel_trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Main trading class
# ----------------------------------------------------------------------
class CitadelQuantumTraderV4:
    """
    Complete integrated trading system with Mission‑Control Dashboard.
    All features (risk limits, levers, exit‑management, etc.) are
    controllable via the dashboard in real‑time.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config: Dict):
        """
        Initialise the trading system.  The base ``config`` can be
        overridden at runtime by the dashboard.

        Args:
            config: Base configuration (may be empty – dashboard will fill)
        """
        self.config = config

        # --------------------------------------------------------------
        # 1️⃣  Initialise Dashboard Connector FIRST
        # --------------------------------------------------------------
        self.dashboard = DashboardConnector()
        self.dashboard.register_config_callback(self.on_config_changed)

        # --------------------------------------------------------------
        # 2️⃣  Apply any configuration that the dashboard already holds
        # --------------------------------------------------------------
        self.apply_dashboard_config()

        # --------------------------------------------------------------
        # 3️⃣  Initialise Risk Management & Execution Engine
        # --------------------------------------------------------------
        self.risk_manager = RiskManagementLayer(self.get_risk_config())
        self.execution_engine = AdvancedExecutionEngine(self.get_execution_config())

        # --------------------------------------------------------------
        # 4️⃣  Statistics container
        # --------------------------------------------------------------
        self.stats = {
            "trades_today": 0,
            "wins_today": 0,
            "losses_today": 0,
            "pl_today": 0.0,
        }

        logger.info("Citadel Quantum Trader v4 initialised with Dashboard")

    # ------------------------------------------------------------------
    # Dashboard‑related helpers
    # ------------------------------------------------------------------
    def apply_dashboard_config(self) -> None:
        """Merge the current dashboard configuration into the main config."""
        dashboard_cfg = self.dashboard.config
        for key, value in dashboard_cfg.items():
            self.config[key] = value
        logger.info("Applied dashboard configuration")

    def on_config_changed(self, new_config: Dict) -> None:
        """
        Callback invoked by the Dashboard when a user changes a setting.

        Args:
            new_config: The freshly‑saved dashboard configuration.
        """
        logger.info("Configuration changed via dashboard")
        for key, value in new_config.items():
            self.config[key] = value

        # Log the most relevant levers for audit purposes
        logger.info("Updated configuration:")
        logger.info(f"  Stacking enabled: {new_config.get('stacking_enabled')}")
        logger.info(f"  R:R ratio: {self.dashboard.get_risk_reward_ratio()}:1")
        logger.info(f"  Breakeven enabled: {new_config.get('breakeven_enabled')}")
        logger.info(f"  Partial‑close enabled: {new_config.get('partial_close_enabled')}")
        logger.info(f"  Trailing‑stop enabled: {new_config.get('trailing_stop_enabled')}")

    # ------------------------------------------------------------------
    # Configuration getters (used by the bot and the risk engine)
    # ------------------------------------------------------------------
    def get_risk_config(self) -> Dict:
        """Assemble the risk‑management configuration."""
        risk_limits = self.dashboard.get_risk_limits()
        return {
            "daily_drawdown_limit_pct": 3.0,
            "weekly_drawdown_limit_pct": 8.0,
            "max_open_positions": risk_limits["max_positions"],
            "emergency_reserve_pct": 5.0,
            "aggressive_pool_pct": 10.0,
            "monitor_interval_seconds": 10,
            "enable_notifications": True,
            "notification_methods": ["telegram"],
        }

    def get_execution_config(self) -> Dict:
        """Assemble the execution‑engine configuration."""
        return {
            "max_slippage_pips": 0.5,
            "enable_liquidity_check": True,
            "min_liquidity_multiple": 2.0,
            "enable_position_scaling": True,
            "scaling_threshold_lots": 50,
            "emergency_capital_reserve_pct": 5.0,
        }

    # ------------------------------------------------------------------
    # Core trading loop (called repeatedly by the outer scheduler)
    # ------------------------------------------------------------------
    def analyze_and_trade(self) -> Optional[Dict]:
        """
        Main trading routine – all levers are honoured via the dashboard.
        Returns the execution result dict if a trade was placed, otherwise None.
        """
# ----------------------------------------------------------------------
# 0️⃣  GLOBAL TRADING‑DAY SCHEDULE
# ----------------------------------------------------------------------
# Define which weekdays the bot is allowed to open new positions.
# 0 = Monday, … , 6 = Sunday  (Python’s datetime.weekday())
TRADING_SCHEDULE = {
    # Example: trade only on the classic “Forex‑active” days.
    # Adjust the list to suit your own business‑hours.
    "allowed_days": [0, 1, 2, 3, 4]   # Mon‑Tue‑Wed‑Thu‑Fri
    # If you ever need a time‑window as well you can add:
    # "start_time": dt_time(5, 0),   # 05:00 UTC
    # "end_time":   dt_time(22, 0), # 22:00 UTC
}


def is_valid_trading_day() -> bool:
    """
    Return ``True`` if **today** is one of the days listed in
    ``TRADING_SCHEDULE['allowed_days']``.  This function can be called
    from the main trading loop to abort the day's processing early.

    If you later want to add a time‑of‑day filter, simply extend the
    function to also compare ``datetime.utcnow().time()`` with the
    optional ``start_time`` / ``end_time`` entries in the schedule.
    """
    today = datetime.utcnow().weekday()          # 0 = Monday … 6 = Sunday
    allowed = TRADING_SCHEDULE.get("allowed_days", [])
    return today in allowed
        # --------------------------------------------------------------
        # 0️⃣  Push current state to the dashboard (visual feedback)
        # --------------------------------------------------------------
        self.update_dashboard()

        # --------------------------------------------------------------
        # 1️⃣  Global risk‑manager pre‑check (position count, kill‑switch)
        # --------------------------------------------------------------
        can_trade, reason = self.risk_manager.can_take_new_trade()
        if not can_trade:
            logger.info(f"Trade blocked by risk manager: {reason}")
            return None

        # --------------------------------------------------------------
        # 2️⃣  Generate a signal (your proprietary confluence engine)
        # --------------------------------------------------------------
        signal = self.analyze_market_for_signals()
        if signal is None:
            return None

        # --------------------------------------------------------------
        # 3️⃣  Apply dashboard‑level filters (confluence, MTF, news, etc.)
        # --------------------------------------------------------------
        if not self.passes_dashboard_filters(signal):
            logger.info("Signal rejected by dashboard filters")
            return None

        # --------------------------------------------------------------
        # 4️⃣  Retrieve the R:R ratio set on the dashboard
        # --------------------------------------------------------------
        rr_ratio = self.dashboard.get_risk_reward_ratio()

        # --------------------------------------------------------------
        # 5️⃣  Stacking logic (if enabled)
        # --------------------------------------------------------------
        if self.dashboard.should_use_stacking():
            current_stack = self.get_current_stack_level(signal["symbol"])
            max_stack = self.dashboard.get_max_stack_level()
            if current_stack >= max_stack:
                logger.info(
                    f"Maximum stack level reached for {signal['symbol']}: "
                    f"{current_stack}/{max_stack}"
                )
                return None

        # --------------------------------------------------------------
        # 6️⃣  Capital allocation (risk manager decides how much is tradable)
        # --------------------------------------------------------------
        acct = mt5.account_info()
        capital_allocation = self.risk_manager.calculate_usable_capital(acct.balance)

        # --------------------------------------------------------------
        # 7️⃣  Execute the trade via the Advanced Execution Engine
        # --------------------------------------------------------------
        result = self.execute_trade(signal, rr_ratio, capital_allocation)

        if result:
            # ----------------------------------------------------------
            # 8️⃣  Set up exit‑management (breakeven, partial‑close, TS)
            # ----------------------------------------------------------
            self.setup_exit_management(result)

            # ----------------------------------------------------------
            # 9️⃣  Update statistics & dashboard
            # ----------------------------------------------------------
            self.stats["trades_today"] += 1
            self.update_dashboard()
            return result

        return None

    # ------------------------------------------------------------------
    # Dashboard filter checks
    # ------------------------------------------------------------------
    def passes_dashboard_filters(self, signal: Dict) -> bool:
        """
        Validate a signal against the live dashboard settings.

        Returns True if the signal satisfies every enabled filter.
        """
        entry_req = self.dashboard.get_entry_requirements()

        # ---- Confluence filter ------------------------------------------------
        if self.dashboard.should_check_confluence():
            if signal["confluence_score"] < entry_req["min_confluence"]:
                logger.info(
                    f"Confluence too low: {signal['confluence_score']} "
                    f"< {entry_req['min_confluence']}"
                )
                return False

        # ---- Entry‑quality filter --------------------------------------------
        if signal.get("entry_quality", 0) < entry_req["min_quality"]:
            logger.info(
                f"Entry quality too low: {signal.get('entry_quality')} "
                f"< {entry_req['min_quality']}"
            )
            return False

        # ---- MTF confirmation -------------------------------------------------
        if self.dashboard.should_check_mtf_confirmation():
            if signal.get("mtf_alignment", 0) < entry_req["mtf_threshold"]:
                logger.info(
                    f"MTF alignment too low: {signal.get('mtf_alignment')} "
                    f"< {entry_req['mtf_threshold']}"
                )
                return False

        # ---- News filter ------------------------------------------------------
        if self.dashboard.should_apply_news_filter():
            if not signal.get("news_clear", True):
                logger.info("Trade blocked by news filter")
                return False

        # ---- Session filter ---------------------------------------------------
        if self.dashboard.should_apply_session_filter():
            if not signal.get("valid_session", True):
                logger.info("Trade blocked by session filter")
                return False

        return True

    # ------------------------------------------------------------------
    # Trade execution (uses the Advanced Execution Engine)
    # ------------------------------------------------------------------
    def execute_trade(
        self,
        signal: Dict,
        rr_ratio: float,
        capital_allocation: Dict,
    ) -> Optional[Dict]:
        """
        Build the full order (SL, TP, lot size) and send it to the
        AdvancedExecutionEngine.

        Returns the raw execution result dict or None on failure.
        """
        # ---- Calculate TP based on the dashboard‑provided R:R -----------------
        sl_pips = signal["sl_pips"]
        tp_pips = sl_pips * rr_ratio
        signal["tp_pips"] = tp_pips

        # ---- Derive TP price -------------------------------------------------
        if signal["direction"] == "BUY":
            signal["tp_price"] = signal["entry_price"] + (tp_pips * 0.0001)
        else:
            signal["tp_price"] = signal["entry_price"] - (tp_pips * 0.0001)

        # ---- Determine lot size (risk‑based) ---------------------------------
        lots, limit_type = self.execution_engine.calculate_safe_lot_size(
            account_balance=capital_allocation["tradable"],
            risk_percentage=self.config.get("risk_per_trade_pct", 2.0),
            stop_loss_pips=sl_pips,
            symbol=signal["symbol"],
            current_leverage=mt5.account_info().leverage,
        )
        if lots == 0:
            logger.warning("Calculated lot size is zero – aborting trade")
            return None

        # ---- Fetch current market price --------------------------------------
        tick = mt5.symbol_info_tick(signal["symbol"])
        entry_price = tick.ask if signal["direction"] == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if signal["direction"] == "BUY" else mt5.ORDER_TYPE_SELL

        # ---- Liquidity check (optional, may down‑size) -----------------------
        sufficient, available = self.execution_engine.check_liquidity(
            symbol=signal["symbol"],
            lots_needed=lots,
            desired_price=entry_price,
            order_type=order_type,
        )
        if not sufficient:
            logger.info(
                f"Liquidity insufficient – reducing lot size from {lots:.2f} to {lots * 0.5:.2f}"
            )
            lots *= 0.5

        # ---- Execute (with optional scaling) ---------------------------------
        results = self.execution_engine.execute_with_scaling(
            symbol=signal["symbol"],
            order_type=order_type,
            total_lots=lots,
            entry_price=entry_price,
            stop_loss=signal["sl_price"],
            take_profit=signal["tp_price"],
            magic_number=self.config["magic_number"],
            comment=f"Citadel_RR{rr_ratio:.0f}_C{signal['confluence_score']}",
        )

        successful = [r for r in results if r["success"]]
        if not successful:
            return None

        return {
            "signal": signal,
            "execution": results,
            "rr_ratio": rr_ratio,
        }

    # ------------------------------------------------------------------
    # Exit‑management (breakeven, partial‑close, trailing‑stop)
    # ------------------------------------------------------------------
    def setup_exit_management(self, trade_result: Dict) -> None:
        """
        Spin up a lightweight monitor thread for a single trade.
        The thread checks the dashboard‑controlled levers and acts
        accordingly (breakeven move, partial close, trailing stop).
        """
        signal = trade_result["signal"]

        def monitor_trade():
            while True:
                try:
                    positions = mt5.positions_get(symbol=signal["symbol"])
                    if not positions:
                        break  # Trade already closed

                    for pos in positions:
                        # ---- Breakeven ----
                        if self.dashboard.should_move_to_breakeven():
                            self.check_breakeven_move(pos, signal)

                        # ---- Partial close ----
                        if self.dashboard.should_partial_close():
                            self.check_partial_close(pos, signal)

                        # ---- Trailing stop ----
                        if self.dashboard.should_use_trailing_stop():
                            self.check_trailing_stop(pos, signal)

                    time.sleep(5)  # poll interval
                except Exception as e:
                    logger.error(f"Exit‑management error: {e}")
                    break

        threading.Thread(target=monitor_trade, daemon=True).start()

    def check_breakeven_move(self, position, signal) -> None:
        """Move stop‑loss to breakeven once the configured R threshold is hit."""
        trigger_r = self.dashboard.get_breakeven_trigger()
        # Profit in pips
        if position.type == mt5.POSITION_TYPE_BUY:
            profit_pips = (position.price_current - position.price_open) / 0.0001
        else:
            profit_pips = (position.price_open - position.price_current) / 0.0001
        profit_r = profit_pips / signal["sl_pips"]
        if profit_r >= trigger_r and position.sl != position.price_open:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": position.symbol,
                "sl": position.price_open,
                "tp": position.tp,
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"✅ Breakeven moved for {position.symbol} #{position.ticket} at {trigger_r:.2f}R"
                )

    def check_partial_close(self, position, signal) -> None:
        """Close a configurable % of the position once the R threshold is hit."""
        params = self.dashboard.get_partial_close_params()
        # Profit in pips
        if position.type == mt5.POSITION_TYPE_BUY:
            profit_pips = (position.price_current - position.price_open) / 0.0001
        else:
            profit_pips = (position.price_open - position.price_current) / 0.0001
        profit_r = profit_pips / signal["sl_pips"]
        if profit_r >= params["trigger_r"]:
            close_vol = position.volume * (params["close_pct"] / 100.0)
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": close_vol,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"PARTIAL_{params['close_pct']}%_at_{profit_r:.1f}R",
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"✅ Partial close ({params['close_pct']}%) for {position.symbol} #{position.ticket} at {profit_r:.1f}R"
                )

    def check_trailing_stop(self, position, signal) -> None:
        """Activate a trailing stop once the configured R threshold is reached."""
        params = self.dashboard.get_trailing_stop_params()
        # Profit in pips
        if position.type == mt5.POSITION_TYPE_BUY:
            profit_pips = (position.price_current - position.price_open) / 0.0001
        else:
            profit_pips = (position.price_open - position.price_current) / 0.0001
        profit_r = profit_pips / signal["sl_pips"]
        if profit_r >= params["start_r"]:
            trail_dist = signal["sl_pips"] * params["distance_r"] * 0.0001
            if position.type == mt5.POSITION_TYPE_BUY:
                new_sl = position.price_current - trail_dist
            else:
                new_sl = position.price_current + trail_dist
            # Only tighten if the new SL is better than the current one
            if (position.type == mt5.POSITION_TYPE_BUY and new_sl > position.sl) or (
                position.type == mt5.POSITION_TYPE_SELL and new_sl < position.sl
            ):
               req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": new_sl,
            "tp": position.tp,
        }

        res = mt5.order_send(req)

        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"✅ Trailing stop updated for {position.symbol} "
                f"#{position.ticket} → new SL {new_sl:.5f}"
            )
        else:
            logger.error(
                f"Failed to update trailing stop for {position.symbol} "
                f"#{position.ticket}: retcode={res.retcode}"
            )
   def get_current_stack_level(self, symbol: str) -> int:
        """Return the number of open positions for *symbol* (used for stacking)."""
        positions = mt5.positions_get(symbol=symbol)
        return len(positions) if positions else 0

    # ------------------------------------------------------------------
    # Placeholder for your actual market‑analysis / signal‑generation logic.
    # ------------------------------------------------------------------
    def analyze_market_for_signals(self) -> Optional[Dict]:
        """
        Run your proprietary confluence scoring system.

        Returns a dictionary with at least the following keys when a
        viable signal is found, otherwise ``None``:

        * ``symbol``          – e.g. ``"EURUSD"``
        * ``direction``      – ``"BUY"`` or ``"SELL"``
        * ``entry_price``    – price at which you would enter
        * ``sl_price``       – stop‑loss price
        * ``sl_pips``        – stop‑loss distance in pips
        * ``confluence_score`` – numeric score used by the dashboard
        * ``entry_quality``  – quality metric (0‑100)
        * ``mtf_alignment``  – percentage (0‑100) from the MTF module
        * any other fields required by your exit‑management logic
        """
        # ------------------------------------------------------------------
        # *** INSERT YOUR SIGNAL ENGINE HERE ***
        # ------------------------------------------------------------------
        # For demonstration we return ``None`` so the bot simply waits.
        return None

    # ------------------------------------------------------------------
    # Dashboard synchronisation helpers
    # ------------------------------------------------------------------
    def update_dashboard(self) -> None:
        """Push the latest internal state to the Mission‑Control dashboard."""
        try:
            dashboard_data = {
                "levers": self.get_lever_status(),
                "platforms": self.get_platform_status(),
                "positions": self.get_positions_data(),
                "performance": self.get_performance_metrics(),
                "risk": self.get_risk_metrics(),
                "activity_log": self.get_recent_activity(),
            }
            self.dashboard.update_dashboard_data(dashboard_data)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to push data to dashboard: {exc}")

    def get_lever_status(self) -> Dict:
        """Return the current status/value of each of the 7 levers."""
        return {
            "entry_quality": {"active": True, "value": 87},
            "market_regime": {"active": True, "value": "TRENDING"},
            "mtf_confirmation": {
                "active": self.dashboard.should_check_mtf_confirmation(),
                "value": "88%",
            },
            "confluence": {
                "active": self.dashboard.should_check_confluence(),
                "value": "4/5",
            },
            "session_filter": {
                "active": self.dashboard.should_apply_session_filter(),
                "value": "LONDON",
            },
            "volatility": {"active": True, "value": "NORMAL"},
            "dynamic_exits": {"active": True, "value": "ACTIVE"},
        }

    def get_platform_status(self) -> Dict:
        """Gather online/offline status and key metrics for each supported platform."""
        enabled = self.dashboard.get_enabled_platforms()
        status = {}
        for platform in ["MT5", "IB", "cTrader", "NinjaTrader", "Tradovate"]:
            if platform not in enabled:
                status[platform] = {"online": False, "balance": 0, "positions": 0, "winrate": 0}
                continue

            if platform == "MT5":
                if mt5.initialize():
                    acc = mt5.account_info()
                    status[platform] = {
                        "online": True,
                        "balance": acc.balance,
                        "positions": mt5.positions_total(),
                        "winrate": 91.5,  # placeholder – compute from history if needed
                    }
                    mt5.shutdown()
                else:
                    status[platform] = {"online": False, "balance": 0, "positions": 0, "winrate": 0}
            else:
                # Stub for other platforms – replace with real API calls
                status[platform] = {"online": False, "balance": 0, "positions": 0, "winrate": 0}
        return status

    def get_positions_data(self) -> List[Dict]:
        """Return a list of dictionaries describing each open position."""
        if not mt5.initialize():
            return []
        raw = mt5.positions_get()
        mt5.shutdown()
        if not raw:
            return []

        positions = []
        for pos in raw:
            positions.append(
                {
                    "symbol": pos.symbol,
                    "direction": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "stack": f"{self.get_current_stack_level(pos.symbol)}/{self.dashboard.get_max_stack_level()}",
                    "entry": pos.price_open,
                    "current": pos.price_current,
                    "pl": pos.profit,
                    "status": "🟢 LIVE",
                }
            )
        return positions

    def get_performance_metrics(self) -> Dict:
        """Aggregate simple performance numbers for the current trading day."""
        total = self.stats["trades_today"]
        wins = self.stats["wins_today"]
        win_rate = (wins / total * 100) if total else 0.0
        return {
            "win_rate": win_rate,
            "trades_today": total,
            "pl_today": self.stats["pl_today"],
            "wins": wins,
            "losses": self.stats["losses_today"],
        }

    def get_risk_metrics(self) -> Dict:
        """Expose the most relevant risk‑management figures."""
        limits = self.dashboard.get_risk_limits()
        used = abs(min(0, self.stats["pl_today"]))
        remaining = limits["daily_loss_limit"] - used
        return {
            "daily_loss_limit": limits["daily_loss_limit"],
            "used": used,
            "remaining": remaining,
            "max_drawdown": limits["max_drawdown_pct"],
            "current_drawdown": 0.0,  # Could be calculated from equity/HWM
        }

    def get_recent_activity(self) -> List[Dict]:
        """Placeholder – in a real system you would pull the last N log lines."""
        # Example structure:
        # [{"timestamp": "...", "message": "..."}]
        return []

# ----------------------------------------------------------------------
# Entry‑point – run the bot in a simple loop (useful for local testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Minimal static configuration – everything else comes from the dashboard
    # ------------------------------------------------------------------
    base_config = {
        "magic_number": 123456,
        "starting_balance": 100_000,
        "trades_taken": 0,
    }

    # ------------------------------------------------------------------
    # Initialise MT5 (required before any MT5 call)
    # ------------------------------------------------------------------
    if not mt5.initialize():
        logger.error("MT5 initialisation failed – exiting")
        exit(1)

    # ------------------------------------------------------------------
    # Instantiate the full trading system
    # ------------------------------------------------------------------
    trader = CitadelQuantumTraderV4(base_config)

    logger.info("=" * 60)
    logger.info("Citadel Quantum Trader v4 – Mission Control Edition")
    logger.info("All features controllable via the dashboard")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Simple endless loop – in production you would run this inside a
    # container orchestrated by your CI/CD pipeline.
    # ------------------------------------------------------------------
    try:
        while True:
            trader.analyze_and_trade()
            time.sleep(60)  # one‑minute pacing; adjust as needed
    except KeyboardInterrupt:
        logger.info("Shutdown signal received – terminating")
    finally:
        mt5.shutdown()

CONFIG_PATH = "/app/config/config.yaml"
NEW_CONFIG_PATH = "/app/config/new_config.yaml"

def reload_config():
    """Called when a new config file appears."""
    # 1️⃣ Replace the old file atomically
    if os.path.isfile(NEW_CONFIG_PATH):
        os.replace(NEW_CONFIG_PATH, CONFIG_PATH)
        # 2️⃣ Signal the main loop to re‑load (e.g., set a flag)
        # Assuming you have a global Config singleton:
        from config_loader import Config
        Config()._load()   # re‑read the file
        print("[WATCHER] Config reloaded from new_config.yaml")
    else:
        print("[WATCHER] No new config found")

def config_watcher(stop_event: threading.Event):
    last_mtime = None
    while not stop_event.is_set():
        try:
            mtime = os.path.getmtime(NEW_CONFIG_PATH)
            if last_mtime is None:
                last_mtime = mtime
            elif mtime != last_mtime:
                reload_config()
                last_mtime = mtime
        except FileNotFoundError:
            # No new config yet – ignore
            pass
        time.sleep(5)   # poll every 5 s

# In your main() function:
stop_evt = threading.Event()
watcher_thread = threading.Thread(target=config_watcher, args=(stop_evt,), daemon=True)
watcher_thread.start()

# Ensure graceful shutdown
def handle_sigterm(signum, frame):
    stop_evt.set()
    watcher_thread.join(timeout=5)
    # … any other cleanup …
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

