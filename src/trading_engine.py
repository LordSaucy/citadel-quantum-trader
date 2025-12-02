#!/usr/bin/env python3
import logging
from typing import Dict
from model_registry import ModelRegistry


from .config_loader import load_config          # your existing config helper
from .feature_extractor import extract_features
from .scorer import get_scorer                  # <-- NEW IMPORT

logger = logging.getLogger(__name__)

def process_bar(bar: Dict) -> None:
    """
    Called for every incoming market bar.
    Handles feature extraction, scoring, and (if appropriate) order placement.
    """
    cfg = load_config()                         # loads config.yaml into a dict

    # ------------------------------------------------------------------
    # 1️⃣  Extract engineered features from the raw bar
    # ------------------------------------------------------------------
    feature_dict = extract_features(bar)

    # ------------------------------------------------------------------
    # 2️⃣  Choose scorer based on config and compute signal strength
    # ------------------------------------------------------------------
    scorer_mode = cfg.get("scorer_mode", "linear")   # "linear" or "lightgbm"
    score_fn = get_scorer(scorer_mode)               # factory call
    signal_strength = score_fn(feature_dict)          # <-- float in [-1, 1]

    logger.info(
        f"Scored bar {bar['symbol']} @ {bar['timestamp']} → "
        f"strength={signal_strength:.3f} (mode={scorer_mode})"
    )

    # ------------------------------------------------------------------
    # 3️⃣  Decision logic (example threshold)
    # ------------------------------------------------------------------
    THRESHOLD = cfg.get("signal_threshold", 0.2)
    if abs(signal_strength) >= THRESHOLD:
        direction = "BUY" if signal_strength > 0 else "SELL"
        # … call your order executor, risk manager, etc.
        logger.info(f"🚀 Emitting {direction} signal for {bar['symbol']}")
        # place_order(direction, bar, ...)

    # else: do nothing – signal too weak

registry = ModelRegistry()

def generate_signal(feature_vec: dict) -> bool:
    # Existing SMC / 7‑lever logic …
    # …
    # Finally combine with the meta‑model vote:
    meta_signal = registry.vote(feature_vec)
    # You can decide to **AND** the two signals (conservative) or **OR** (aggressive)
    return meta_signal and smc_signal   # conservative default

def evaluate_signal(market_data, dom_df):
    # Existing SMC checks …
    smc_ok = smc_filter(market_data)

    # New LIR feature
    lir = compute_lir(dom_df)

    # Simple heuristic: require |LIR| > 0.3 for a signal to be considered
    # (you can make this a tunable parameter in config.yaml)
    lir_ok = abs(lir) > 0.30

    # Confluence score – you already have a weighted sum of levers.
    # Add LIR as an extra lever with weight w_lir (e.g., 0.15).
    confluence_score = (
        w_price_action * price_action_score +
        w_ema_cross    * ema_cross_score +
        w_atr_break    * atr_break_score +
        w_lir          * (abs(lir) - 0.30)   # normalized to 0‑1
    )

    # Final decision
    return smc_ok and lir_ok and confluence_score >= config["GRADE_A_PLUS"]
