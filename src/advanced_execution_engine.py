import time
from typing import Dict, Any

from .config import logger
from .risk_management import RiskManagementLayer
from prometheus_client import Histogram, Counter
from smc import SMCEngine
engine = SMCEngine()          # created at start‑up
from config_loader import Config
from volatility_breakout import is_expanding
from data_fetcher import get_recent_history   # existing helper that returns a DataFrame
from .venue_manager import VenueManager
from trade_logger import TradeLogger   # import the logger singleton or instantiate it
from shock_detector import should_block_trade
from regime_ensemble import match_regime, current_regime_vector
from garch_vol import forecast_vol   # you already have a GARCH helper
import asyncio
from src.telemetry import trace
import os
import json
import time
import logging
from pathlib import Path

from prometheus_client import Counter, Gauge



from .guard_helpers import (
    check_depth,
    check_latency,
    check_spread,
    check_volatility,
)



# -------------------------------------------------
# advanced_execution_engine.py
# -------------------------------------------------
import logging
from datetime import datetime
import pytz

# Existing imports …
from src.market_data_manager import MarketDataManager   # provides latest tick/depth
from src.risk_management_layer import RiskManagementLayer
from src.bot_control import BotControl                 # wrapper around pause/resume/kill‑switch
from src.hmm_regime import HMMRegime

# ---- NEW: import the guard classes ----------------
from src.guards import (
    SentimentGuard,
    CalendarLockout,
    VolatilityGuard,
    ShockDetector,
)

# Initialise once (e.g., at bot start)
risk_mgr = RiskManager(db=session)   # pass your DB/session object

# ----------------------------------------------------------------------
# Stubbed broker adapters – replace with real MT5 / IBKR SDK calls
# ----------------------------------------------------------------------
# Existing histogram (you already have this)
order_latency_seconds = Histogram(
    "order_latency_seconds",
    "Latency from order submission to broker ACK (seconds)",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
)

# New counter to capture price‑difference after fill
order_price_slippage = Counter(
    "order_price_slippage",
    "Absolute price difference between submitted price and fill price (pips)",
    ["symbol", "side"],
)

# Assuming you have a global logger instance:
LOGGER = TradeLogger()   # or however you obtain it in your code base

# Somewhere in src/advanced_execution_engine.py (or any module that needs a regime)
# ------------------------------------------------------------------
# Initialise once at process start (e.g. in the global scope)
# ------------------------------------------------------------------
cfg = Config().settings
regime_classifier = HMMRegime(cfg)

# Try to load a pre‑trained model; if it fails we automatically fall back
if not regime_classifier.load():
    log.warning("Running with fallback regime detector (no HMM model)")

# ------------------------------------------------------------------
# Inside the signal‑processing loop
# ------------------------------------------------------------------
def process_bar(bar: dict):
    # Assume `bar` already contains the engineered features required by the HMM
    regime = regime_classifier.predict(bar)

    # You can now use `regime` to gate the rest of the pipeline:
    if regime == 2:          # high‑volatility regime → be extra cautious
        risk_multiplier = 0.5
    elif regime == 1:
        risk_multiplier = 1.0
    else:                    # low‑volatility regime
        risk_multiplier = 1.5

    # Pass `risk_multiplier` downstream to the position‑sizing logic
    ...

    async def _pre_trade_guard(self, signal) -> bool:
        """
        Returns True if *all* safety checks pass.
        `signal` contains at least: symbol, volume (positive for buy, negative for sell),
        and any other fields you need for the guard (e.g., required_volume).
        """
        # 1️⃣ Depth / LIR
        if not check_depth(self.broker,
                           symbol=signal["symbol"],
                           required_volume=abs(signal["volume"]),
                           min_lir=0.5):
            self.logger.warning("Trade skipped – depth/LIR guard")
            return False

        # 2️⃣ Latency
        if not check_latency(self.broker, max_latency_sec=0.15):
            self.logger.warning("Trade skipped – latency guard")
            return False

        # 3️⃣ Spread / slippage
        if not check_spread(self.broker,
                            symbol=signal["symbol"],
                            max_spread_pips=0.5):
            self.logger.warning("Trade skipped – spread guard")
            return False

        # 4️⃣ Volatility‑spike (ATR)
        if not check_volatility(self.tech_calc,
                                symbol=signal["symbol"],
                                atr_multiplier=2.0,
                                max_atr_pct=0.20):
            self.logger.warning("Trade skipped – volatility‑spike guard")
            return False

        # All checks passed
        return True

    async def execute_trade(self, signal):
        """
        Entry point called by the signal engine.
        """
        # -----------------------------------------------------------------
        # 0️⃣ Quick sanity: make sure the bot is not paused / kill‑switch
        # -----------------------------------------------------------------
        if self.state.is_paused or self.state.kill_switch_active:
            self.logger.info("Bot paused / kill‑switch active – ignoring signal")
            return

        # -----------------------------------------------------------------
        # 1️⃣ Run the predictive‑control guard
        # -----------------------------------------------------------------
        if not await self._pre_trade_guard(signal):
            # Guard already logged the reason; we just return.
            return

        # -----------------------------------------------------------------
        # 2️⃣ Normal execution path (size calc, risk check, send order)
        # -----------------------------------------------------------------
        stake = self.risk_manager.compute_stake(...)
        # … existing order construction …
        self.broker.send_order(...)


def _check_liquidity(self, symbol, required_volume, side):
    depth = self.broker.get_market_depth(symbol, depth=30)
    total_bid = sum(d['bid_vol'] for d in depth)
    total_ask = sum(d['ask_vol'] for d in depth)

    # Need at least 2× required volume on the side we intend to trade
    available = total_bid if side == "buy" else total_ask
    if required_volume * 2 > available:
        # ---- 1️⃣ Alert the operator (already present) ----
        img_path = plot_depth_heatmap(symbol, depth)
        send_alert(
            title="⚠ Liquidity warning",
            severity="warning",
            details={"symbol": symbol,
                     "required_vol": required_volume,
                     "heatmap": img_path},
        )

        # ---- 2️⃣ Record a synthetic loss (new) ----
        # bucket_id is available on the engine (self.bucket_id)
        LOGGER.log_synthetic_loss(
            bucket_id=self.bucket_id,
            symbol=symbol,
            volume=required_volume,
            reason="Depth < 2× required volume"
        )

        # Abort the trade
        return False
    return True

# Example in the order‑placement flow
if not self._check_liquidity(symbol, required_volume, side):
    # The function already logged the loss; just skip the trade.
    return  # early exit – trade is not sent to the broker



cfg = Config().settings
RR_TARGET = cfg.get("RR_target", 5.0)   # default 5 if missing

  # -----------------------------------------------------------------
    # Helper: compute lot size from stake and contract_notional
    # -----------------------------------------------------------------
    def _lot_from_stake(self, stake: float) -> float:
        contract_notional = self.cfg.get("contract_notional", 100_000)  # $ per lot
        return stake / contract_notional




def build_order(entry_price: float, risk_amount: float):
    # risk_amount = 1 R (the amount you are willing to lose)
    tp_price = entry_price + RR_TARGET * risk_amount
    sl_price = entry_price - 1.0 * risk_amount   # 1 R loss
    return {"tp": tp_price, "sl": sl_price}


def _watch_config(self):
    # existing file‑watch loop …
    if mtime != last_mtime:
        # reload the whole config dict
        cfg = Config().settings
        # propagate to SMC
        engine.refresh_params()
        last_mtime = mtime

 def __init__(self, cfg, redis_client):
        self.cfg = cfg
        self.risk = RiskManagementLayer(cfg)
        self.control = BotControl()               # expose pause/resume/kill
        self.market = MarketDataManager(cfg)
     self.venue_mgr = VenueManager()   

        # ---- instantiate each guard -----------------
        self.sentiment_guard = SentimentGuard(cfg, redis_client)
        self.calendar_lockout = CalendarLockout(cfg, __import__("economic_calendar"))
        self.volatility_guard = VolatilityGuard(cfg)
        self.shock_detector = ShockDetector(cfg)

    # -------------------------------------------------
    # Core method – called for every generated signal
    # -------------------------------------------------
    def process_signal(self, signal):
        """
        `signal` is the output of your 7‑lever SMC + regime engine.
        It should contain at least:
            - symbol, side (buy/sell), price, timestamp, quantity
        """

        
        # -------------------------------------------------
        # 1️⃣ Run the unified shock‑detector
        # -------------------------------------------------
        blocked, reason = should_block_trade(symbol)
        if blocked:
            self.logger.warning(
                f"⚡️ Trade for {symbol} BLOCKED by shock‑detector ({reason})"
            )
            # Record the block in the immutable ledger
            from trade_logger import TradeLogger
            TradeLogger().log_event(
                {
                    "event": "trade_blocked",
                    "symbol": symbol,
                    "reason": reason,
                    "timestamp": time.time(),
                }
            )
            # Increment a Prometheus gauge (optional)
            from prometheus_client import Counter
            blocked_counter = Counter(
                "trade_blocked_reason_total",
                "Blocked trades by reason",
                ["reason"],
            )
            blocked_counter.labels(reason=reason).inc()
            return  # **Exit early – no order will be sent**

         -------------------------------------------------
        # 2️⃣ Normal processing (regime filter, SMC, etc.)
        # -------------------------------------------------
        if not self.regime_filter.allows(signal):
            return  # regime rejected – already logged elsewhere

        # ... existing risk‑check, position‑stacking, etc.
        self.execute_trade(signal)

            
        # -----------------------------------------------------------------
        # 1️⃣ Calendar lock‑out (high‑impact macro events)
        # -----------------------------------------------------------------
        if self.calendar_lockout.is_locked():
            log.info("[Guard] Calendar lock‑out active – skipping signal")
            return None   # signal discarded

        # -----------------------------------------------------------------
        # 2️⃣ News‑sentiment guard
        # -----------------------------------------------------------------
        if not self.sentiment_guard.check():
            log.info("[Guard] Sentiment guard rejected signal")
            return None

        # -----------------------------------------------------------------
        # 3️⃣ Volatility‑spike guard
        # -----------------------------------------------------------------
        # Assume you have a method that returns the latest ATR(14)
        current_atr = self.market.get_atr(period=14)
        if not self.volatility_guard.check(current_atr):
            log.info("[Guard] Volatility spike – rejecting signal")
            return None

        # -----------------------------------------------------------------
        # 4️⃣ Shock detector (spread, desync, liquidity)
        # -----------------------------------------------------------------
        snapshot = self.market.get_latest_snapshot()
        if not self.shock_detector.check(snapshot):
            log.info("[Guard] Shock detector rejected signal")
            return None

        # -----------------------------------------------------------------
        # 5️⃣ If we made it here, the signal is “clean” – continue with
        #    risk sizing, order creation, and broker submission.
        # -----------------------------------------------------------------
        stake = self.risk.compute_stake(signal)   # existing method
        order = self.build_order(signal, stake)   # your existing logic

        # Send to broker (MT5, IBKR, etc.)
        result = self.broker.send_order(order)

        # Record outcome, update ledger, emit Prometheus metrics…
        self.handle_execution_result(result, signal)

        return result


class MT5Gateway:
    def __init__(self, host: str, login: str, password: str):
        self.host = host
        self.login = login
        self.password = password
        logger.info("MT5Gateway configured for %s", host)

    def send_order(self, symbol: str, side: str, qty: float, price: float) -> bool:
        logger.debug("MT5 order %s %s %.4f @ %.5f", side, symbol, qty, price)
        # TODO: integrate MetaTrader5 API here
        return True


class IBKRGateway:
    def __init__(self, host: str, api_key: str, secret: str):
        self.host = host
        self.api_key = api_key
        self.secret = secret
        logger.info("IBKRGateway configured for %s", host)

    def can_enter_trade(equity, other_context):
    # 1️⃣ Dynamic kill‑switch
    if risk_mgr.check_dynamic_kill_switch(equity):
        # Signal the higher‑level loop to pause new entries
        return False

    # 2️⃣ Existing checks (regime, volatility, etc.)
    ...

    if depth_ok:
    depth_ok_gauge.labels(bucket_id=signal.bucket_id,
                          symbol=signal.symbol).inc()
else:
    depth_ok_gauge.labels(bucket_id=signal.bucket_id,
                          symbol=signal.symbol).dec()


    def send_order(self, symbol: str, side: str, qty: float, price: float) -> bool:
        logger.debug("IBKR order %s %s %.4f @ %.5f", side, symbol, qty, price)
        # TODO: integrate ib_insync / REST API here
        return True

def generate_signal(bar, previous_bar):
    # 1️⃣ Get recent history for ATR calculation (e.g., last 30 bars)
    hist = get_recent_history(symbol=bar['symbol'], length=30)   # returns DataFrame

    # 2️⃣ Volatility breakout test
    if not is_expanding(bar, previous_bar, hist):
        logger.debug("Volatility breakout filter rejected signal")
        return None

    # 3️⃣ Continue with existing regime‑forecast, etc.
    ...

# ----------------------------------------------------------------------
# Main execution engine – orchestrates risk checks, confluence scores,
# and finally sends the order to the appropriate broker.
# ----------------------------------------------------------------------
class AdvancedExecutionEngine:
    def __init__(self, risk: RiskManagementLayer, confluence) -> None:
        self.risk = risk
        self.confluence = confluence
        self.venue_mgr = VenueManager()   

        # Load broker credentials from env (via src.config.env helper)
        from .config import env

        self.mt5 = MT5Gateway(
            host=env("METATRADER5_HOST", "mt5-gateway.internal"),
            login=env("METATRADER5_LOGIN", ""),
            password=env("METATRADER5_PASSWORD", ""),
        )
        self.ibkr = IBKRGateway(
            host=env("IBKR_HOST", "ibkr-gateway.internal"),
            api_key=env("IBKR_API_KEY", ""),
            secret=env("IBKR_SECRET", ""),
        )

    if cfg.get("mode") == "arb_only":
    # `arb_signal` is a dict with the three legs and the raw profit estimate
    try:
        await triangular_arb_executor.execute_triangular_arb(
            broker=self.broker,
            legs=arb_signal["legs"],
            gross_profit_pips=arb_signal["gross_profit_pips"],
        )
        logger.info("✅ Triangular arb executed successfully")
    except ArbExecutionError as exc:
        logger.warning(f"⚠️ Arb aborted: {exc}")
        # Optionally record a metric so you can see how often the guard fires
        metrics.arb_guard_hits.inc()
        # Continue with normal flow (skip this arb)
        continue


    # ------------------------------------------------------------------
    def _choose_gateway(self, symbol: str) -> Any:
        """Very simple routing – you can make this smarter."""
        # Example heuristic: if symbol ends with “USD” use MT5, else IBKR
        return self.mt5 if symbol.endswith("USD") else self.ibkr

    # ------------------------------------------------------------------
    def execute_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        current_open_positions: int,
        equity: float,
        high_water_mark: float,
    ) -> Dict[str, Any]:
        """
        Public entry point used by the REST API / internal scheduler.
        Returns a dict that can be JSON‑encoded for the caller.
        """
        # 1️⃣  Position‑limit check
        ok, reason = self.risk.can_take_new_trade(current_open_positions)
        if not ok:
            return {"executed": False, "reason": reason}

        # 2️⃣  Draw‑down / kill‑switch check
        self.risk.evaluate_drawdown(equity, high_water_mark)
        if self.risk.kill_switch_active:
            return {
                "executed": False,
                "reason": f"KILL‑SWITCH: {self.risk.kill_reason}",
            }

        # 3️⃣  Confluence score – simple placeholder
        # (In production you would call self.confluence.calculate_score(...))
        confluence_score = self.confluence.get_current_score()
        if confluence_score < 70:  # arbitrary threshold
            return {
                "executed": False,
                "reason": f"Confluence score too low ({confluence_score})",
            }

        # 4️⃣  Send order to the selected broker
        gateway = self._choose_gateway(symbol)
        success = gateway.send_order(symbol, side, qty, price)

        if success:
            logger.info("Order executed: %s %s %.4f @ %.5f", side, symbol, qty, price)
            return {"executed": True, "reason": "order placed"}
        else:
            logger.error("Broker rejected order %s %s", side, symbol)
            return {"executed": False, "reason": "broker rejection"}

 def send_order(self, symbol, volume, side, price=None, sl=None, tp=None, comment=""):
        # 1️⃣ Record *submission* timestamp and price
        submit_ts = time.time()
        submit_price = price if price is not None else self.market_price(symbol)

        # 2️⃣ Send the order to the broker (still async / blocking as before)
        order_id = self.broker.send_order(
            symbol=symbol,
            volume=volume,
            side=side,
            price=price,
            sl=sl,
            tp=tp,
            comment=comment,
        )

        # 3️⃣ Wait for the broker’s ACK/fill callback (you probably already have a
        #    `on_order_filled(order_id, fill_price, fill_ts)` hook somewhere).
        #    Hook into that callback to compute latency & slippage.
        return order_id, submit_ts, submit_price
 

new_equity = self.account.get_equity()   # whatever method you have
self.risk_manager.calculate_usable_capital(bucket_id=self.bucket_id,
                                            equity=new_equity)

def on_order_filled(self, order_id, fill_price, fill_timestamp):
    # Retrieve the stored submission info (you can keep a dict keyed by order_id)
    sub = self.submissions = {"submit_ts": submit_ts,
    "submit_price": submit_price,
    "symbol": symbol,
    "side": side,
}  # key: order_id → {"submit_ts":…, "submit_price":…, "symbol":…, "side":…}

    if not sub:
        # No submission record – maybe a manual order; just log and return
        self.logger.warning(f"Fill received for unknown order {order_id}")
        return

    # ① Compute latency
    latency = fill_timestamp - sub["submit_ts"]
    order_latency_seconds.observe(latency)

    # ② Compute price slippage (absolute difference, expressed in pips)
    #     For FX, 1 pip = 0.0001 (or 0.01 for JPY pairs). Adjust as needed.
    pip_factor = 0.0001
    if sub["symbol"].endswith("JPY"):
        pip_factor = 0.01
    slippage = abs(fill_price - sub["submit_price"]) / pip_factor
    order_price_slippage.labels(symbol=sub["symbol"], side=sub["side"]).inc(slippage)

    # ③ Log for audit
    self.logger.info(
        f"Order {order_id} filled – latency={latency:.3f}s, slippage={slippage:.1f} pips"
    )
   # -----------------------------------------------------------------
    # Core order submission – now with multi‑venue depth guard
    # -----------------------------------------------------------------
    def send_order(self, signal):
        """
        signal contains: symbol, side ('buy'/'sell'), price, etc.
        """
        # 1️⃣ Compute the stake & lot size
        equity = self.risk.get_current_equity(signal.bucket_id)
        stake = equity * self.risk.compute_stake(signal.bucket_id, equity)  # $ risk amount
        lot = self._lot_from_stake(stake)

        # 2️⃣ Query depth from all venues
        agg = self.venue_mgr.aggregate_depth(signal.symbol, depth=20)

        # 3️⃣ Verify minimum depth
        depth_ok = self.venue_mgr.meets_minimum(lot, agg)

        # 4️⃣ Log the depth result (will appear in the immutable ledger)
        self.risk.log_depth_check(
            bucket_id=signal.bucket_id,
            symbol=signal.symbol,
            depth_ok=depth_ok,
            agg_bid=agg["bid_volume"],
            agg_ask=agg["ask_volume"],
        )

        if not depth_ok:
            # We could either abort completely or shrink the lot.
            # For safety we abort and let the edge‑decay detector handle it.
            self.risk.record_skip(signal, reason="insufficient_depth")
            return {"status": "rejected", "reason": "insufficient_depth"}

        # 5️⃣ Proceed with the normal execution path (now we know depth is ok)
        order = self.broker.send_order(
            symbol=signal.symbol,
            volume=lot,
            side=signal.side,
            price=signal.price,
            sl=signal.sl,
            tp=signal.tp,
        )
        # … rest of the existing logic (record trade, update ledger, etc.) …
        return order
        
def generate_signal(bar):
    # 1️⃣ Existing LSTM/HMM predictions
    lstm_probs = lstm_model.predict(bar_features)          # shape (3,)
    hmm_state = hmm_model.predict(bar_features)            # int 0/1/2

    # 2️⃣ GARCH volatility forecast (next minute)
    recent_returns = compute_returns(bar_history)          # pandas Series
    garch_res = garch_fit(recent_returns)                 # arch model result
    vol_forecast = forecast_vol(garch_res, steps=1)[0]     # scalar variance

    # 3️⃣ Build the ensemble vector & match
    vec = current_regime_vector(lstm_probs, hmm_state, vol_forecast)
    cluster_id, similarity = match_regime(vec)

    # 4️⃣ Decision: allow trade only if we matched a *good* historical regime
    if cluster_id is None:
        logger.info(f"[REGIME] No matching historical regime (sim={similarity:.2f}) – signal suppressed")
        return None   # abort this candidate

    # Optional: you can also read the stored performance of the matched cluster
    # and adjust the risk‑fraction dynamically:
    meta = META[cluster_id]
    if meta["win_rate"] < 0.92:   # safety net
        logger.warning(f"[REGIME] Matched cluster {cluster_id} has low win‑rate ({meta['win_rate']:.2%}) – lowering risk")
        risk_modifier = 0.5   # halve the risk‑fraction for this trade
    else:
        risk_modifier = 1.0

    # 5️⃣ Continue with the existing 7‑lever SMC pipeline,
    # passing `risk_modifier` downstream (e.g., via the RiskManagementLayer)
    signal = build_signal_from_features(bar, lstm_probs, hmm_state, vol_forecast)
    signal.risk_modifier = risk_modifier
    return signal


def compute_stake(bucket_id: int, equity: float, risk_modifier: float = 1.0):
    trade_idx = get_trade_counter(bucket_id) + 1
    f = RISK_SCHEDULE.get(trade_idx, RISK_SCHEDULE.get("default", 0.40))
    f *= risk_modifier   # <-- new line
    stake = risk_manager.allocate_for_trade(bucket_id, symbol, equity, risk_frac)
    return stake 

if cfg['risk_controls'].get('use_lir', False):
    if not depth_guard.check_lir(order_book):
        logger.info("LIR guard rejected trade")
        return RejectCode.LIQUIDITY_IMBALANCE

if result.retcode == mt5.TRADE_RETCODE_DONE:
    pnl = (exit_price - entry_price) * volume * RR_target   # calculate profit in USD
    risk_manager.record_trade_result(bucket_id, pnl)

async def execution_engine(execution_queue):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("execution_engine"):
        while True:
            signal = await execution_queue.get()
            with tracer.start_as_current_span("send_order"):
                # Existing order‑submission code (MT5 API, etc.)
                await send_order_to_broker(signal)

log = logging.getLogger(__name__)

class AdvancedExecutionEngine:
    def __init__(self, shadow_mode: bool = False):
        self.shadow_mode = shadow_mode
        # Load credentials from Vault (unchanged)
        self._load_credentials()
        # Initialise Prometheus metrics (shared with paper‑trading)
        self.order_counter = Counter(
            "cqt_orders_total",
            "Total number of orders the engine would have sent",
            ["symbol", "direction", "shadow"]
        )
        self.latency_gauge = Gauge(
            "cqt_order_latency_seconds",
            "Round‑trip latency of order preparation (seconds)",
            ["symbol", "shadow"]
        )
        self.slippage_gauge = Gauge(
            "cqt_order_slippage_pips",
            "Absolute slippage of the would‑be fill (pips)",
            ["symbol", "shadow"]
        )
        # Shadow log file (rotated daily by systemd/journald if you like)
        self.shadow_log_path = Path("/var/log/cqt/shadow.log")
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Public API used by the bot
    # -----------------------------------------------------------------
    def send_order(self, order):
        """
        `order` is a dict with at least:
        - symbol
        - direction ("BUY"/"SELL")
        - price (requested price)
        - volume
        - stop_loss, take_profit
        """
        start = time.time()

        # -----------------------------------------------------------------
        # 1️⃣  Build the payload that would be sent to the broker
        # -----------------------------------------------------------------
        payload = {
            "symbol": order["symbol"],
            "direction": order["direction"],
            "price": order["price"],
            "volume": order["volume"],
            "stop_loss": order["stop_loss"],
            "take_profit": order["take_profit"],
        }

        # -----------------------------------------------------------------
        # 2️⃣  If we are in SHADOW mode, **don’t call the broker**.
        # -----------------------------------------------------------------
        if self.shadow_mode:
            # Simulate a “fill” price – we use the requested price plus a tiny
            # random jitter to mimic market movement (you can make this more
            # sophisticated if you wish).
            import random
            jitter = random.uniform(-0.0005, 0.0005)   # ±0.5 pip for FX
            fill_price = order["price"] + jitter

            # Compute slippage in pips (absolute)
            pip_size = self._pip_size(order["symbol"])
            slippage = abs(fill_price - order["price"]) / pip_size

            # Record metrics
            latency = time.time() - start
            self.latency_gauge.labels(symbol=order["symbol"], shadow="yes").set(latency)
            self.slippage_gauge.labels(symbol=order["symbol"], shadow="yes").set(slippage)
            self.order_counter.labels(symbol=order["symbol"],
                                      direction=order["direction"],
                                      shadow="yes").inc()

            # Write a structured JSON line to the shadow log
            log_entry = {
                "timestamp": time.time(),
                "symbol": order["symbol"],
                "direction": order["direction"],
                "requested_price": order["price"],
                "fill_price": round(fill_price, 5),
                "volume": order["volume"],
                "slippage_pips": round(slippage, 3),
                "latency_seconds": round(latency, 4),
                "note": "SHADOW – order NOT sent to broker"
            }
            with self.shadow_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            log.debug(f"🕶️  Shadow order logged: {log_entry}")
            # Return a *fake* success response that mimics the real broker API
            return {
                "success": True,
                "fill_price": fill_price,
                "order_id": f"shadow-{int(time.time()*1000)}",
                "reason": "shadow"
            }

        # -----------------------------------------------------------------
        # 3️⃣  REAL‑MODE – call the broker (unchanged from your existing code)
        # -----------------------------------------------------------------
        # Example for MT5 – replace with your actual implementation
        response = self._mt5_send_order(payload)   # <-- existing low‑level call
        latency = time.time() - start
        self.latency_gauge.labels(symbol=order["symbol"], shadow="no").set(latency)
        self.order_counter.labels(symbol=order["symbol"],
                                  direction=order["direction"],
                                  shadow="no").inc()

        # Compute slippage if the broker returns a fill price
        if response.get("fill_price"):
            pip_size = self._pip_size(order["symbol"])
            slippage = abs(response["fill_price"] - order["price"]) / pip_size
            self.slippage_gauge.labels(symbol=order["symbol"], shadow="no").set(slippage)

        return response

    # -----------------------------------------------------------------
    # Helper – pip size per symbol (same as in costs module)
    # -----------------------------------------------------------------
    def _pip_size(self, symbol: str) -> float:
        from .metrics import PIP_VALUE   # already in your repo
        return PIP_VALUE.get(symbol, 0.0001)
def _load_adapter(broker_type: str, cfg: dict):
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

class AdvancedExecutionEngine:
    def __init__(self):
        cfg = Config().settings
        broker_type = cfg.get("broker_type", "mt5").lower()
        self.broker = _load_adapter(broker_type, cfg)
        if not self.broker.connect():
            raise RuntimeError("Failed to connect to broker")

class MultiBrokerRouter:
    def __init__(self, primary: BrokerInterface, secondary: BrokerInterface,
                 latency_threshold: float = 0.20):
        self.primary = primary
        self.secondary = secondary
        self.latency_threshold = latency_threshold  # seconds

    def send_order(self, **kwargs):
        # 1️⃣ Submit to primary and record timestamp
        start_ts = time.time()
        try:
            result = self.primary.send_order(**kwargs)
            # The primary will later call back with fill info.
            # We store the start_ts so the fill callback can compute latency.
            result["submit_ts"] = start_ts
            return result
        except Exception as exc:
            # Primary failed outright (e.g., network error)
            logger.warning(f"Primary broker error: {exc}; falling back")
            return self._fallback(**kwargs, start_ts=start_ts)

    def _fallback(self, **kwargs, start_ts):
        # 2️⃣ Try secondary
        try:
            result = self.secondary.send_order(**kwargs)
            result["submit_ts"] = start_ts
            result["fallback"] = True
            return result
        except Exception as exc2:
            logger.error(f"Both brokers failed: {exc2}")
            raise  # propagate up – the bot will treat it as a hard failure

    # Optional: a helper that can be called from the fill‑callback
    # to decide *after* the fact whether latency was too high.
    def maybe_fallback_based_on_latency(self, order_id, latency):
        if latency > self.latency_threshold:
            logger.info(f"Latency {latency:.3f}s > {self.latency_threshold}s – "
                        f"retrying order {order_id} on secondary")
            # Re‑issue the same order on secondary (you need to keep the original
            # order parameters somewhere – e.g., in a dict keyed by order_id)
            params = self._original_params[order_id]
            return self.secondary.send_order(**params)
        return None

