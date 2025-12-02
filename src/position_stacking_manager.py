#!/usr/bin/env python3
"""
Position Stacking Manager

Integrates stacking logic with the 7‑lever optimization system.

* Base positions must satisfy all 7 levers (≥ 99.7 % win‑rate setups)
* Stack positions are pull‑back entries that compound profit
* Max 4 positions per symbol (1 base + up to 3 stacks)
* Total risk per symbol capped at 10 % of account equity
* 5‑minute cooldown between successive stacks
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from .utils.heatmap import plot_depth_heatmap
from .alerting import send_alert   # you already have a Slack/PagerDuty helper
# make sure you import the broker (self.broker is already set)


# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class StackTrade:
    """Individual trade belonging to a stack."""
    ticket: int
    symbol: str
    direction: str                     # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_amount: float
    entry_time: datetime
    stack_level: int                   # 1 = base, 2‑4 = stacks
    parent_ticket: Optional[int] = None
    breakeven_moved: bool = False
    partial_closed: bool = False
    profit: float = 0.0                # floating P/L (updated each tick)


@dataclass
class StackGroup:
    """All positions for a single symbol."""
    symbol: str
    direction: str                     # BUY or SELL
    base_ticket: int
    stack_tickets: List[int]           # tickets of stack positions
    total_risk: float                  # sum of risk amounts (USD)
    total_profit: float                # realised profit
    accumulated_profit: float          # realised + floating profit
    created_at: datetime
    last_stack_time: datetime
    quality_score: float               # 7‑lever quality (0‑100)


# ----------------------------------------------------------------------
# Controller – holds mutable parameters, Prometheus gauges and a tiny API
# ----------------------------------------------------------------------
from prometheus_client import Gauge
from flask import Flask, jsonify, request, abort
import threading
import time
import os

class StackingController:
    """
    Runtime‑tunable parameters for the stacking manager.

    * Stored in a JSON file (mounted volume) → survives restarts.
    * Exported as Prometheus gauges → visible on Grafana.
    * Tiny Flask API (GET/POST) → Grafana UI can modify them live.
    """

    DEFAULTS = {
        # ----- limits -------------------------------------------------
        "max_positions_per_symbol": 4,
        "max_total_risk_percent": 10.0,          # % of account equity
        "cooldown_between_stacks_sec": 300,      # 5 min
        "min_profit_to_stack": 200.0,            # USD
        # ----- risk multipliers per stack level ----------------------
        "stack_risk_multiplier_1": 1.0,          # base
        "stack_risk_multiplier_2": 1.5,
        "stack_risk_multiplier_3": 2.0,
        "stack_risk_multiplier_4": 2.5,
        # ----- profit multiplier for sizing next stack -------------
        "stack_profit_multiplier": 1.5,
        # ----- R‑multiple thresholds --------------------------------
        "breakeven_r": 1.0,
        "partial_exit_r": 2.0,
        "target_r_min": 3.0,
        "target_r_max": 5.0,
        # ----- debug flag --------------------------------------------
        "debug": False,
    }

    CONFIG_PATH = Path("/app/config/stacking_config.json")   # mount this dir

    # Prometheus gauge (one per key)
    _gauge: Dict[str, Gauge] = {}

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._load_or_create()
        self._register_gauges()
        self._start_watcher()
        self._start_api()

    # ------------------------------------------------------------------
    # Load persisted JSON or fall back to defaults
    # ------------------------------------------------------------------
    def _load_or_create(self) -> None:
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    self.values = json.load(f)
                logger.info(f"StackingController – loaded config from {self.CONFIG_PATH}")
            except Exception as exc:   # pragma: no cover
                logger.error(f"Failed to read config – using defaults ({exc})")
                self.values = self.DEFAULTS.copy()
        else:
            logger.info("StackingController – no config file, using defaults")
            self.values = self.DEFAULTS.copy()
            self._persist()

    # ------------------------------------------------------------------
    # Persist to JSON
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(self.values, f, indent=2)
        except Exception as exc:   # pragma: no cover
            logger.error(f"Could not persist stacking config: {exc}")

    # ------------------------------------------------------------------
    # Register Prometheus gauges (one per key)
    # ------------------------------------------------------------------
    def _register_gauges(self) -> None:
        for key, val in self.values.items():
            g = Gauge(
                "stacking_parameter",
                "Runtime‑tunable stacking parameter",
                ["parameter"],
            )
            g.labels(parameter=key).set(val)
            self._gauge[key] = g

    # ------------------------------------------------------------------
    # Update a single key (used by API and file‑watcher)
    # ------------------------------------------------------------------
    def set(self, key: str, value: float) -> None:
        if key not in self.values:
            raise KeyError(f"Unknown stacking parameter: {key}")
        self.values[key] = float(value)
        self._gauge[key].labels(parameter=key).set(float(value))
        self._persist()
        logger.info(f"StackingController – set {key} = {value}")

    # ------------------------------------------------------------------
    # Read a key
    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    # ------------------------------------------------------------------
    # File watcher – reload if the JSON is edited externally
    # ------------------------------------------------------------------
    def _start_watcher(self) -> None:
        def _watch():
            last_mtime = (
                self.CONFIG_PATH.stat().st_mtime
                if self.CONFIG_PATH.exists()
                else 0
            )
            while not self._stop.is_set():
                if self.CONFIG_PATH.exists():
                    mtime = self.CONFIG_PATH.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info("StackingController – config file changed, reloading")
                        self._load_or_create()
                        for k, v in self.values.items():
                            self._gauge[k].labels(parameter=k).set(v)
                        last_mtime = mtime
                time.sleep(2)

        threading.Thread(target=_watch, daemon=True, name="stacking-config-watcher").start()

    # ------------------------------------------------------------------
    # Tiny Flask API (GET /config, POST /config/<key>)
    # ------------------------------------------------------------------
    def _start_api(self) -> None:
        app = Flask(__name__)

        @app.route("/config", methods=["GET"])
        def get_all():
            return jsonify(self.values)

        @app.route("/config/<key>", methods=["GET"])
        def get_one(key: str):
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            return jsonify({key: self.values[key]})

        @app.route("/config/<key>", methods=["POST", "PUT", "PATCH"])
        def set_one(key: str):
            if key not in self.values:
                abort(404, description=f"Parameter {key} not found")
            try:
                payload = request.get_json(force=True)
                if not payload or "value" not in payload:
                    abort(400, description="JSON body must contain 'value'")
                self.set(key, payload["value"])
                return jsonify({key: self.values[key]})
            except Exception as exc:   # pragma: no cover
                logger.error(f"API error while setting {key}: {exc}")
                abort(500, description=str(exc))

        @app.route("/healthz", methods=["GET"])
        def health():
            return "OK", 200

        def _run():
            # expose on all interfaces; Grafana will call http://<container>:5005
            app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)

        threading.Thread(target=_run, daemon=True, name="stacking-flask-api").start()

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()


# ----------------------------------------------------------------------
# Global controller instance (importable from other modules)
# ----------------------------------------------------------------------
stacking_controller = StackingController()


# ----------------------------------------------------------------------
# Position Stacking Manager – core business logic
# ----------------------------------------------------------------------
class PositionStackingManager:
    """
    Manages position stacking with profit compounding.

    Rules (mirroring the doc‑string at the top of the file):

    1️⃣  Max 4 positions per symbol (1 base + up to 3 stacks)
    2️⃣  Each stack uses 1‑2× profit from previous trades for sizing
    3️⃣  Independent SL per position
    4️⃣  Max 10 % total risk across all stacks for a symbol
    5️⃣  5‑minute cooldown between successive stacks
    6️⃣  Pull‑back entries must be in the direction of the established trend
    """

    # ------------------------------------------------------------------
    # Constructor – initialise internal trackers
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.active_trades: Dict[int, StackTrade] = {}          # ticket → trade
        self.stack_groups: Dict[str, StackGroup] = {}          # symbol → group
        self.closed_trades: List[StackTrade] = []
        self.stacking_enabled = True

        logger.info("Position Stacking Manager initialised")
        logger.info(f"  Max positions per symbol: {stacking_controller.get('max_positions_per_symbol')}")
        logger.info(f"  Max total risk %: {stacking_controller.get('max_total_risk_percent')}%")
        logger.info(f"  Cooldown between stacks: {stacking_controller.get('cooldown_between_stacks_sec')} s")

    # ------------------------------------------------------------------
    # 1️⃣  Base position handling
    # ------------------------------------------------------------------
    def place_base_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        risk_percent: float,
        quality_score: float,
    ) -> Optional[int]:
        """
        Place the **base** position (stack level 1).

        Returns the MT5 ticket or ``None`` on failure.
        """
        # ------------------------------------------------------------------
        # Guard – one base per symbol only
        # ------------------------------------------------------------------
        if symbol in self.stack_groups:
            logger.warning(f"Base position already exists for {symbol}")
            return None

        # ------------------------------------------------------------------
        # Account info & risk calculation
        # ------------------------------------------------------------------
        account = mt5.account_info()
        if not account:
            logger.error("Failed to obtain MT5 account info")
            return None

        balance = account.balance
        risk_amount = balance * (risk_percent / 100.0)

        # ------------------------------------------------------------------
        # Lot size calculation
        # ------------------------------------------------------------------
        lot_size = self._calculate_lot_size(symbol, entry_price, stop_loss, risk_amount)
        if lot_size <= 0:
            logger.error(f"Lot size calculation failed for {symbol}")
            return None

        # ------------------------------------------------------------------
        # Take‑profit based on random R‑target within configured bounds
        # ------------------------------------------------------------------
        risk = abs(entry_price - stop_loss)
        target_r = np.random.uniform(
            stacking_controller.get("target_r_min"),
            stacking_controller.get("target_r_max")
        )
        take_profit = (
            entry_price + risk * target_r
            if direction.upper() == "BUY"
            else entry_price - risk * target_r
        )
        order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL

        # ------------------------------------------------------------------
        # Submit the order to MT5
        # ------------------------------------------------------------------
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": f"BASE_{direction}_Q{int(quality_score)}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Base order failed for {symbol}: {result.comment}")
            return None

        # ------------------------------------------------------------------
        # Record the trade & create the stack group
        # ------------------------------------------------------------------
        trade = StackTrade(
            ticket=result.order,
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            risk_amount=risk_amount,
            entry_time=datetime.now(),
            stack_level=1,
        )
        self.active_trades[result.order] = trade

        self.stack_groups[symbol] = StackGroup(
            symbol=symbol,
            direction=direction.upper(),
            base_ticket=result.order,
            stack_tickets=[],
            total_risk=risk_amount,
            total_profit=0.0,
            accumulated_profit=0.0,
            created_at=datetime.now(),
            last_stack_time=datetime.now(),
            quality_score=quality_score,
        )

        logger.info(f"✅ BASE {direction.upper()} opened for {symbol}")
        logger.info(f"  Ticket: {result.order}")
        logger.info(f"  Entry: {entry_price:.5f}")
        logger.info(f"  SL: {stop_loss:.5f}")
        logger.info(f"  TP: {take_profit:.5f}")
        logger.info(f"  Size: {lot_size:.2f} lots")
        logger.info(f"  Risk: ${risk_amount:.2f} ({risk_percent:.2f} %)")
        logger.info(f"  Quality Score: {quality_score:.1f}/100")

        return result.order

    # ------------------------------------------------------------------
    # 2️⃣  Stack eligibility checks
    # ------------------------------------------------------------------
    def can_add_stack(self, symbol: str) -> Tuple[bool, str]:
        """
        Verify whether a new stack can be added for ``symbol``.

        Returns ``(True, "reason")`` if allowed, otherwise ``(False, reason)``.
        """
        if not self.stacking_enabled:
            return False, "Stacking disabled"

        if symbol not in self.stack_groups:
            return False, "No base position exists"

        group = self.stack_groups[symbol]

        # Max positions per symbol (base + stacks)
        if len(group.stack_tickets) >= stacking_controller.get("max_positions_per_symbol") - 1:
            return False, f"Maximum stacks ({stacking_controller.get('max_positions_per_symbol')}) reached"

        # Cool‑down enforcement
        elapsed = (datetime.now() - group.last_stack_time).total_seconds()
        if elapsed < stacking_controller.get("cooldown_between_stacks_sec"):
            remaining = int(stacking_controller.get("cooldown_between_stacks_sec") - elapsed)
            return False, f"Cooldown active – {remaining}s remaining"

        # Need enough accumulated profit to fund the next stack
        if group.accumulated_profit < stacking_controller.get("min_profit_to_stack"):
            need = stacking_controller.get("min_profit_to_stack") - group.accumulated_profit
            return False, f"Insufficient profit (${need:.2f} needed)"

        # Total risk cap (percentage of account equity)
        account = mt5.account_info()
        if account:
            total_risk_pct = (group.total_risk / account.balance) * 100.0
            if total_risk_pct >= stacking_controller.get("max_total_risk_percent"):
                return False, f"Total risk limit ({total_risk_pct:.2f} %) exceeded"

        return True, "Stack approved"

    # ------------------------------------------------------------------
    # 3️⃣  Place a stack position
    # ------------------------------------------------------------------
    def place_stack_position(self, symbol: str, entry_price: float, stop_loss: float) -> Optional[int]:
        """
        Add a new stack position for ``symbol`` using accumulated profit
        to determine risk size.

        Returns the MT5 ticket or ``None`` on failure.
        """
        can_stack, reason = self.can_add_stack(symbol)
        if not can_stack:
            logger.warning(f"Cannot stack {symbol}: {reason}")
            return None

        group = self.stack_groups[symbol]

        # --------------------------------------------------------------
        # Determine stack level (base = 1, first stack = 2, etc.)
        # --------------------------------------------------------------
        stack_level = len(group.stack_tickets) + 2   # +2 because base is level 1

        # --------------------------------------------------------------
        # Risk amount for this stack
        # --------------------------------------------------------------
        base_risk = group.accumulated_profit * stacking_controller.get("stack_profit_multiplier")
        level_multiplier = stacking_controller.get(f"stack_risk_multiplier_{stack_level}")
        stack_risk = base_risk * level_multiplier

        # Cap risk to a sensible max (5 % of account per position)
        account = mt5.account_info()
        if account:
            max_risk = account.balance * 0.05
            stack_risk = min(stack_risk, max_risk)

        # --------------------------------------------------------------
        # Lot size
        # --------------------------------------------------------------
        lot_size = self._calculate_lot_size(symbol, entry_price, stop_loss, stack_risk)
        if lot_size <= 0:
            logger.error(f"Invalid lot size for stack on {symbol}")
            return None

        # --------------------------------------------------------------
        # Take‑profit (random R‑target)
        # --------------------------------------------------------------
        risk = abs(entry_price - stop_loss)
        target_r = np.random.uniform(
            stacking_controller.get("target_r_min"),
            stacking_controller.get("target_r_max")
        )
        if group.direction == "BUY":
            take_profit = entry_price + risk * target_r
            order_type = mt5.ORDER_TYPE
       else:
            take_profit = entry_price - risk * target_r
            order_type = mt5.ORDER_TYPE_SELL

        # --------------------------------------------------------------
        # Submit the stack order to MT5
        # --------------------------------------------------------------
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": f"STACK{stack_level}_{group.direction}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Stack order failed for {symbol}: {result.comment}")
            return None

        # --------------------------------------------------------------
        # Record the new stack trade
        # --------------------------------------------------------------
        trade = StackTrade(
            ticket=result.order,
            symbol=symbol,
            direction=group.direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            risk_amount=stack_risk,
            entry_time=datetime.now(),
            stack_level=stack_level,
            parent_ticket=group.base_ticket,
        )

        self.active_trades[result.order] = trade
        group.stack_tickets.append(result.order)
        group.total_risk += stack_risk
        group.last_stack_time = datetime.now()

        logger.info(f"🔥 STACK {stack_level} opened for {symbol}")
        logger.info(f"  Ticket: {result.order}")
        logger.info(f"  Entry: {entry_price:.5f}")
        logger.info(f"  SL: {stop_loss:.5f}")
        logger.info(f"  TP: {take_profit:.5f}")
        logger.info(f"  Size: {lot_size:.2f} lots")
        logger.info(f"  Risk: ${stack_risk:.2f} (derived from profit)")
        logger.info(f"  Total positions for {symbol}: {self.get_stack_count(symbol)}")
        logger.info(f"  Accumulated profit: ${group.accumulated_profit:.2f}")

        return result.order

    # ------------------------------------------------------------------
    # 4️⃣  Ongoing position management (breakeven moves, partial exits)
    # ------------------------------------------------------------------
    def manage_open_positions(self) -> None:
        """
        Iterate over all open MT5 positions and apply the following rules:

        • Move stop‑loss to breakeven once the position hits 1 R.
        • Close 50 % of the position once it reaches 2 R (partial profit).
        • Update floating profit and propagate it to the corresponding
          ``StackGroup`` (so the next stack can use the updated profit).
        """
        positions = mt5.positions_get()
        if not positions:
            return

        for pos in positions:
            ticket = pos.ticket
            if ticket not in self.active_trades:
                continue

            trade = self.active_trades[ticket]

            # Current R‑multiple
            current_r = self._calculate_r_multiple(pos)

            # Update floating profit (used for group accumulation)
            trade.profit = pos.profit
            self._update_group_profit(trade.symbol)

            # ---- Breakeven at 1 R -------------------------------------------------
            if current_r >= stacking_controller.get("breakeven_r") and not trade.breakeven_moved:
                if self._modify_stop_loss(ticket, pos.price_open):
                    trade.breakeven_moved = True
                    logger.info(
                        f"🔒 {trade.symbol} #{ticket} moved SL to breakeven "
                        f"({current_r:.2f} R)"
                    )

            # ---- Partial exit at 2 R ---------------------------------------------
            if current_r >= stacking_controller.get("partial_exit_r") and not trade.partial_closed:
                profit = self._close_partial(ticket, 0.5)   # close 50 %
                if profit > 0:
                    trade.partial_closed = True
                    group = self.stack_groups.get(trade.symbol)
                    if group:
                        group.total_profit += profit
                        group.accumulated_profit += profit
                    logger.info(
                        f"💰 {trade.symbol} #{ticket} 50 % closed at "
                        f"{current_r:.2f} R – profit ${profit:.2f}"
                    )

    # ------------------------------------------------------------------
    # 5️⃣  Detect closed positions and clean up internal state
    # ------------------------------------------------------------------
    def check_closed_positions(self) -> None:
        """Move closed MT5 tickets from ``active_trades`` to ``closed_trades``."""
        current_tickets = {p.ticket for p in mt5.positions_get() or []}
        closed_tickets = [t for t in self.active_trades if t not in current_tickets]

        for ticket in closed_tickets:
            # Retrieve the historic deal(s) for this ticket
            deals = mt5.history_deals_get(position=ticket)
            total_profit = sum(d.profit for d in deals) if deals else 0.0

            trade = self.active_trades[ticket]
            logger.info(f"📊 {trade.symbol} #{ticket} closed – profit ${total_profit:.2f}")

            # Update the associated StackGroup
            if trade.symbol in self.stack_groups:
                group = self.stack_groups[trade.symbol]

                # Base position closed → remove whole group if no stacks remain
                if ticket == group.base_ticket:
                    if not group.stack_tickets or all(
                        st not in self.active_trades for st in group.stack_tickets
                    ):
                        logger.info(f"🏁 Stack group {trade.symbol} fully closed")
                        logger.info(f"  Total profit: ${group.total_profit + total_profit:.2f}")
                        del self.stack_groups[trade.symbol]
                # Stack ticket closed → remove from list
                elif ticket in group.stack_tickets:
                    group.stack_tickets.remove(ticket)

                group.total_profit += total_profit
                group.accumulated_profit += total_profit

            # Archive the trade and purge from active dict
            self.closed_trades.append(trade)
            del self.active_trades[ticket]

    # ------------------------------------------------------------------
    # 6️⃣  Helper utilities
    # ------------------------------------------------------------------
    def _calculate_lot_size(
        self,
        symbol: str,
        entry: float,
        stop_loss: float,
        risk_amount: float,
    ) -> float:
        """Convert a dollar risk amount into a lot size respecting broker limits."""
        info = mt5.symbol_info(symbol)
        if not info:
            return 0.0

        point = info.point
        # Pip‑risk (in points)
        pip_risk = abs(entry - stop_loss) / point
        tick_value = info.trade_tick_value
        if tick_value == 0:
            return 0.0

        lot = risk_amount / (pip_risk * tick_value)

        # Round to broker step size
        lot = round(lot / info.volume_step) * info.volume_step
        lot = max(info.volume_min, min(lot, info.volume_max))
        return lot

    def _calculate_r_multiple(self, position) -> float:
        """Return the current R‑multiple of a live position."""
        entry = position.price_open
        sl = position.sl
        cur = position.price_current
        risk = abs(entry - sl)
        if risk == 0:
            return 0.0

        if position.type == mt5.ORDER_TYPE_BUY:
            return (cur - entry) / risk
        else:
            return (entry - cur) / risk

    def _modify_stop_loss(self, ticket: int, new_sl: float) -> bool:
        """Move the stop‑loss of an open position."""
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def _close_partial(self, ticket: int, fraction: float = 0.5) -> float:
        """
        Close a fraction of a position and return the realised profit.

        ``fraction`` is a value between 0 and 1.
        """
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return 0.0
        pos = pos[0]

        volume = pos.volume * fraction
        info = mt5.symbol_info(pos.symbol)
        if volume < info.volume_min:
            volume = info.volume_min

        order_type = (
            mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        )
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Partial_2R",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return 0.0

        # Approximate profit (price delta * volume * contract size)
        contract = info.trade_contract_size
        if pos.type == mt5.ORDER_TYPE_BUY:
            profit = (price - pos.price_open) * volume * contract
        else:
            profit = (pos.price_open - price) * volume * contract
        return profit

    def _update_group_profit(self, symbol: str) -> None:
        """Refresh the floating profit of a group (used for sizing next stack)."""
        if symbol not in self.stack_groups:
            return
        group = self.stack_groups[symbol]

        floating = 0.0
        for tkt in [group.base_ticket] + group.stack_tickets:
            if tkt in self.active_trades:
                floating += self.active_trades[tkt].profit

        group.accumulated_profit = group.total_profit + floating

    # ------------------------------------------------------------------
    # 7️⃣  Status / reporting helpers
    # ------------------------------------------------------------------
    def get_stack_count(self, symbol: str) -> int:
        """Return the total number of positions (base + stacks) for a symbol."""
        grp = self.stack_groups.get(symbol)
        return 1 + len(grp.stack_tickets) if grp else 0

    def get_status(self) -> Dict:
        """Snapshot of the manager’s current state (useful for dashboards)."""
        return {
            "stacking_enabled": self.stacking_enabled,
            "active_trades": len(self.active_trades),
            "active_groups": len(self.stack_groups),
            "closed_trades": len(self.closed_trades),
            "groups": {
                sym: {
                    "direction": grp.direction,
                    "positions": self.get_stack_count(sym),
                    "total_risk_usd": grp.total_risk,
                    "accumulated_profit_usd": grp.accumulated_profit,
                    "quality_score": grp.quality_score,
                }
                for sym, grp in self.stack_groups.items()
            },
        }

    def enable_stacking(self) -> None:
        self.stacking_enabled = True
        logger.info("✅ Stacking ENABLED")

    def disable_stacking(self) -> None:
        self.stacking_enabled = False
        logger.info("⚠️ Stacking DISABLED")

# ----------------------------------------------------------------------
# Global singleton – importable from ``src/main.py``
# ----------------------------------------------------------------------
stacking_manager = PositionStackingManager()

class PositionStackingManager:
    ...

    def _check_liquidity(self, symbol: str, required_volume: float, side: str) -> bool:
        """
        Returns True if there is enough depth on the requested side.
        If not, generates a heat‑map PNG and fires an alert.
        `side` must be either "buy" or "sell".
        """
        depth = self.broker.get_market_depth(symbol, depth=30)

        # Aggregate total volume on each side
        total_bid = sum(item["bid_vol"] for item in depth)
        total_ask = sum(item["ask_vol"] for item in depth)

        # Simple rule: we want at least 2× the required volume on the side we will hit
        if side == "buy":
            available = total_bid
        elif side == "sell":
            available = total_ask
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        if required_volume * 2 > available:
            # Not enough liquidity – generate a heat‑map for post‑mortem / alert
            try:
                img_path = plot_depth_heatmap(symbol, depth)
                alert_payload = {
                    "title": "⚠ Liquidity warning",
                    "severity": "warning",
                    "details": {
                        "symbol": symbol,
                        "required_volume": required_volume,
                        "available_volume": available,
                        "heatmap_path": img_path,
                    },
                }
                send_alert(**alert_payload)
            except Exception as exc:
                # If the heat‑map itself fails we still want to abort the trade
                send_alert(
                    title="⚠ Liquidity warning (heat‑map failed)",
                    severity="warning",
                    details={
                        "symbol": symbol,
                        "required_volume": required_volume,
                        "available_volume": available,
                        "error": str(exc),
                    },
                )
            return False

        # Sufficient depth – proceed
        return True

# src/position_stacking_manager.py
MEDIAN_ATR = cfg.get("median_atr", 0.001)   # 0.1 % of price, set after warm‑up
VOL_THRESHOLD_MULT = cfg.get("vol_threshold_mult", 2.0)

def max_allowed_stacks(symbol: str, current_atr: float) -> int:
    """Return 3 by default, but shrink to 2 when volatility spikes."""
    if current_atr > VOL_THRESHOLD_MULT * MEDIAN_ATR:
        return 2
    return 3
