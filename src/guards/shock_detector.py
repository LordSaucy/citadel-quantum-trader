import logging
from prometheus_client import Counter, Gauge

log = logging.getLogger("citadel.shock_detector")

shock_hits = Counter(
    "shock_detector_hits_total",
    "Trades rejected by the shock detector",
    ["type"]  # "spread", "desync", "liquidity"
)

class ShockDetector:
    """
    Three independent checks that run on the freshest market snapshot.
    All thresholds are configurable via `config.yaml`.
    """

    def __init__(self, cfg):
        """
        cfg – expects:
            shock_detector:
                enabled: true
                spread_multiplier: 3.0   # current spread > 3× avg_spread → reject
                max_tick_age_secs: 2      # quote older than 2 s → reject
                lir_threshold: 0.6        # LIR > 0.6 → reject
                min_depth: 0.02           # minimum depth as a fraction of lot size
        """
        self.enabled = cfg.get("shock_detector", {}).get("enabled", True)
        self.spread_mul = cfg["shock_detector"].get("spread_multiplier", 3.0)
        self.max_age = cfg["shock_detector"].get("max_tick_age_secs", 2)
        self.lir_thr = cfg["shock_detector"].get("lir_threshold", 0.6)
        self.min_depth = cfg["shock_detector"].get("min_depth", 0.02)

    # -----------------------------------------------------------------
    # Helper: compute average spread (you probably already have this in
    # market_data_manager – we just reuse it)
    # -----------------------------------------------------------------
    @staticmethod
    def _avg_spread(spread_series):
        # `spread_series` is a list of recent spreads (float)
        if not spread_series:
            return 0.0
        return sum(spread_series) / len(spread_series)

    # -----------------------------------------------------------------
    # Public entry point – returns True if the trade may proceed
    # -----------------------------------------------------------------
    def check(self, market_snapshot) -> bool:
        """
        `market_snapshot` is a dict that you already build when you receive
        a new tick / depth update. Expected keys:
            - bid, ask, bid_volume, ask_volume
            - timestamp (datetime, UTC)
            - spread_history (list[float]) – recent spreads
            - depth (float) – total volume on the best‑price level
        """
        if not self.enabled:
            return True

        # 1️⃣ Spread explosion
        cur_spread = market_snapshot["ask"] - market_snapshot["bid"]
        avg_spread = self._avg_spread(market_snapshot.get("spread_history", []))
        if avg_spread > 0 and cur_spread > avg_spread * self.spread_mul:
            shock_hits.labels(type="spread").inc()
            log.warning("[ShockDetector] Spread explosion: %.6f > %.6f×avg",
                        cur_spread, self.spread_mul)
            return False

        # 2️⃣ Quote desync (old tick)
        age = (datetime.datetime.utcnow().replace(tzinfo=pytz.UTC) -
               market_snapshot["timestamp"]).total_seconds()
        if age > self.max_age:
            shock_hits.labels(type="desync").inc()
            log.warning("[ShockDetector] Quote desync – age %.2f s > %d s",
                        age, self.max_age)
            return False

        # 3️⃣ Liquidity vacuum (LIR & depth)
        bid_vol = market_snapshot.get("bid_volume", 0)
        ask_vol = market_snapshot.get("ask_volume", 0)
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            lir = 0.0
        else:
            lir = (
