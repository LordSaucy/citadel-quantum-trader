def should_trade(regime_label: str, other_factors: dict) -> bool:
    # Existing 7‑lever SMC check ...
    if not smc_filter(other_factors):
        return False

    # NEW guard: reject trades when regime is bear AND sentiment is strongly negative
    if regime_label == 'bear' and other_factors['sentiment_score'] < -0.5:
        logger.info("Skipping trade – bear regime + negative sentiment")
        return False

    # OPTIONAL: dynamically lower risk_fraction when sentiment is mildly negative
    if other_factors['sentiment_score'] < 0.0:
        risk_adj = max(0.5, 1.0 + other_factors['sentiment_score'])   # e.g., -0.3 → 0.7
        other_factors['risk_multiplier'] = risk_adj
    return True
