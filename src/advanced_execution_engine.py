import time
from typing import Dict, Any

from .config import logger
from .risk_management import RiskManagementLayer

# ----------------------------------------------------------------------
# Stubbed broker adapters – replace with real MT5 / IBKR SDK calls
# ----------------------------------------------------------------------
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
