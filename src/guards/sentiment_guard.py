import logging
from prometheus_client import Counter, Gauge

log = logging.getLogger("citadel.sentiment_guard")

# Prometheus metrics (optional but handy for alerts)
sentiment_guard_hits = Counter(
    "sentiment_guard_hits_total",
    "Number of times the news‑sentiment guard rejected a signal",
    ["reason"]  # e.g. "high_sentiment", "negative_sentiment"
)

class SentimentGuard:
    """
    Checks a pre‑computed sentiment score (‑1 … +1) against configurable thresholds.
    The score is expected to be stored in the shared Redis key `sentiment:latest`
    by the sentiment‑ingestor container (see PDF 1).
    """

    def __init__(self, cfg, redis_client):
        """
        cfg – dict from config.yaml, expects:
            sentiment:
                guard_enabled: true
                positive_threshold: 0.30   # > 0.30 ⇒ overly bullish → reject
                negative_threshold: -0.30  # < -0.30 ⇒ overly bearish → reject
        redis_client – a redis.StrictRedis instance (same as used by the bot)
        """
        self.enabled = cfg.get("sentiment", {}).get("guard_enabled", True)
        self.pos_thr = cfg["sentiment"].get("positive_threshold", 0.30)
        self.neg_thr = cfg["sentiment"].get("negative_threshold", -0.30)
        self.redis = redis_client

    def _latest_score(self) -> float | None:
        """Read the latest sentiment score from Redis (float or None)."""
        raw = self.redis.get("sentiment:latest")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def check(self) -> bool:
        """
        Returns True if the trade **may proceed**, False if it must be rejected.
        """
        if not self.enabled:
            return True

        score = self._latest_score()
        if score is None:
            # No sentiment data – be permissive (or you could reject)
            return True

        if score > self.pos_thr:
            sentiment_guard_hits.labels(reason="high_sentiment").inc()
            log.info("[SentimentGuard] Rejecting – score %.3f > %.3f (bullish)",
                     score, self.pos_thr)
            return False
        if score < self.neg_thr:
            sentiment_guard_hits.labels(reason="low_sentiment").inc()
            log.info("[SentimentGuard] Rejecting – score %.3f < %.3f (bearish)",
                     score, self.neg_thr)
            return False

        return True
