import os
import time
import logging
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from prometheus_client import Counter, Gauge

log = logging.getLogger(__name__)

class BinanceFuturesAdapter:
    """
    Minimal Binance Futures wrapper that conforms to the CQT adapter contract.
    It uses the REST API for price quotes and the signed POST endpoint for orders.
    """

    def __init__(self, paper_mode: bool = False):
        self.paper_mode = paper_mode
        self._load_credentials()
        self._init_metrics()
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.creds["api_key"]})

    # -----------------------------------------------------------------
    # 1️⃣  Credential loading (Vault secret)
    # -----------------------------------------------------------------
    def _load_credentials(self):
        """
        Expected secret shape (JSON stored in Vault):
        {
            "api_key": "<binance api key>",
            "api_secret": "<binance secret>",
            "base_url": "https://fapi.binance.com"
        }
        """
        # The generic Vault loader lives in src/vault_helper.py – reuse it
        from .vault_helper import get_secret
        secret_path = os.getenv("VAULT_SECRET_PATH")
        secret = get_secret(secret_path)   # returns a dict
        self.creds = {
            "api_key": secret["api_key"],
            "api_secret": secret["api_secret"],
            "base_url": secret.get("base_url", "https://fapi.binance.com")
        }

    # -----------------------------------------------------------------
    # 2️⃣  Prometheus metrics (same names as other adapters)
    # -----------------------------------------------------------------
    def _init_metrics(self):
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

    # -----------------------------------------------------------------
    # 3️⃣  Helper – sign Binance request (HMAC SHA256)
    # -----------------------------------------------------------------
    def _signed_payload(self, params: dict) -> dict:
        import hmac, hashlib, urllib.parse, time
        ts = int(time.time() * 1000)
        params["timestamp"] = ts
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.creds["api_secret"].encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    # -----------------------------------------------------------------
    # 4️⃣  Get latest price (used for slippage calculation)
    # -----------------------------------------------------------------
    def get_quote(self, symbol: str) -> float:
        url = urljoin(self.creds["base_url"], "/fapi/v1/ticker/price")
        resp = self.session.get(url, params={"symbol": symbol})
        resp.raise_for_status()
        return float(resp.json()["price"])

    # -----------------------------------------------------------------
    # 5️⃣  Send order (market order for simplicity)
    # -----------------------------------------------------------------
    def send_order(self, order: dict) -> dict:
        """
        order dict (same shape as other adapters):
        {
            "symbol": "BTCUSDT",
            "direction": "BUY" or "SELL",
            "price": float (requested price, used only for slippage calc),
            "quantity": float (in contracts),
            "stop_loss": float,
            "take_profit": float,
        }
        """
        start = time.time()

        # -----------------------------------------------------------------
        # 5.1  Build Binance order payload (market order)
        # -----------------------------------------------------------------
        side = "BUY" if order["direction"].upper() == "BUY" else "SELL"
        payload = {
            "symbol": order["symbol"],
            "side": side,
            "type": "MARKET",
            "quantity": order["quantity"],
            # Binance Futures requires `newClientOrderId` for idempotency – optional
        }
        signed = self._signed_payload(payload)

        # -----------------------------------------------------------------
        # 5.2  Call the API
        # -----------------------------------------------------------------
        url = urljoin(self.creds["base_url"], "/fapi/v1/order")
        try:
            resp = self.session.post(url, params=signed, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            success = True
            fill_price = float(data["avgPrice"])
            reject_code = None
        except Exception as exc:
            log.warning("Binance order failed: %s", exc)
            success = False
            fill_price = None
            reject_code = getattr(exc, "response", None)
            if reject_code is not None:
                reject_code = reject_code.status_code

        # -----------------------------------------------------------------
        # 5.3  Record metrics (latency, slippage, counter)
        # -----------------------------------------------------------------
        latency = time.time() - start
        self.latency_gauge.labels(symbol=order["symbol"],
                                  shadow="yes" if self.paper_mode else "no").set(latency)
        self.order_counter.labels(symbol=order["symbol"],
                                  direction=order["direction"],
                                  shadow="yes" if self.paper_mode else "no").inc()

        if success:
            # Slippage = |fill – requested| expressed in pips
            pip_size = self._pip_size(order["symbol"])
            slippage = abs(fill_price - order["price"]) / pip_size
            self.slippage_gauge.labels(symbol=order["symbol"],
                                       shadow="yes" if self.paper_mode else "no").set(slippage)

        # -----------------------------------------------------------------
        # 5.4  Return unified response dict (same shape as MT5/IBKR)
        # -----------------------------------------------------------------
        return {
            "success": success,
            "fill_price": fill_price,
            "order_id": data.get("orderId") if success else None,
            "reject_code": reject_code,
        }

    # -----------------------------------------------------------------
    # 6️⃣  Helper – pip size (Binance futures use 0.01 for most contracts)
    # -----------------------------------------------------------------
    def _pip_size(self, symbol: str) -> float:
        # Very simple mapping – you can expand with a JSON file if needed.
        if symbol.endswith("USDT"):
            return 0.01   # e.g., BTCUSDT price moves in 0.01 increments
        return 0.0001   # fallback for very small‑price assets

    # -----------------------------------------------------------------
    # 7️⃣  Close / cleanup (optional)
    # -----------------------------------------------------------------
    def close(self):
        self.session.close()
