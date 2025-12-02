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

cfg = Config().settings
RR_TARGET = cfg.get("RR_target", 5.0)   # default 5 if missing

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
    sub = self.submissions.pop(order_id, None)
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

