from market_data.lir import compute_lir
from market_data.dom_cache import DomCache

def build_feature_vector(self, symbol: str, bucket_id: int) -> dict:
    # Existing feature extraction …
    features = {
        "mtf_alignment": self._calc_mtf(symbol),
        "head_and_shoulders": self._calc_hands(symbol),
        # …
    }

    # -----------------------------------------------------------------
    # NEW – LIR and total depth
    # -----------------------------------------------------------------
    dom_df = DomCache().get(symbol)
    lir = compute_lir(dom_df)
    total_depth = dom_df["bid_volume"].sum() + dom_df["ask_volume"].sum()
    features["lir"]   = lir
    features["depth"] = total_depth
    return features


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
