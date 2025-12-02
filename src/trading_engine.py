#!/usr/bin/env python3
import logging
from typing import Dict

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
