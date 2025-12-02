#!/usr/bin/env python3
"""
Signal pipeline – decides whether a raw market bar should become a trading signal.

The pipeline runs a series of pre‑filters (volatility breakout, news overlay,
etc.) that you already have elsewhere in the code‑base.  The final step
performed here is the **Multi‑TF confirmation** implemented in
`src/multi_tf.py`.

If the confirmation fails, the function returns ``None`` – the caller can
simply ignore the bar.  Otherwise it returns a dictionary (or any object
your downstream code expects) that represents a valid signal.
"""

import logging
from typing import Any, Dict, Optional

# ----------------------------------------------------------------------
# Imports from the rest of the CQT code‑base
# ----------------------------------------------------------------------
from .multi_tf import confirm_signal_across_tf   # <-- the function you already wrote
# You will also import your existing pre‑filter helpers here, e.g.:
# from .volatility_breakout import is_vol_breakout
# from .news_overlay import passes_news_filter
# from .regime_forecast import forecast_regime
# from .risk_management import evaluate_risk

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def generate_signal(bar: Dict[str, Any],
                    previous_bar: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Run the full signal‑generation pipeline for a single market bar.

    Parameters
    ----------
    bar : dict
        The current OHLCV bar (must contain at least ``symbol`` and ``timestamp``).
    previous_bar : dict or None, optional
        The previous bar – useful for some filters that need a look‑back.

    Returns
    -------
    dict or None
        A populated signal dict if **all** checks pass, otherwise ``None``.
    """

    # --------------------------------------------------------------
    # 1️⃣  Existing pre‑filters (volatility breakout, news overlay, …)
    # --------------------------------------------------------------
    # Example placeholders – replace with your real functions:
    # if not is_vol_breakout(bar, previous_bar):
    #     logger.debug("Volatility breakout filter failed – discarding signal")
    #     return None
    #
    # if not passes_news_filter(bar):
    #     logger.debug("News overlay filter failed – discarding signal")
    #     return None

    # --------------------------------------------------------------
    # 2️⃣  Multi‑TF confirmation (the new piece you asked about)
    # --------------------------------------------------------------
    if not confirm_signal_across_tf(bar["symbol"], bar["timestamp"]):
        logger.debug("Multi‑TF confirmation failed – discarding signal")
        return None

    # --------------------------------------------------------------
    # 3️⃣  Regime forecast, risk evaluation, etc.
    # --------------------------------------------------------------
    # Example placeholders – replace with your real logic:
    # regime = forecast_regime(bar)
    # risk_ok = evaluate_risk(bar, regime)
    # if not risk_ok:
    #     logger.debug("Risk evaluation failed – discarding signal")
    #     return None

    # --------------------------------------------------------------
    # 4️⃣  Build the final signal payload
    # --------------------------------------------------------------
    signal: Dict[str, Any] = {
        "symbol": bar["symbol"],
        "timestamp": bar["timestamp"],
        "price": bar["close"],          # or whatever price you use for entry
        "direction": "BUY",             # or derive from your upstream logic
        # You can add any extra fields your downstream engine expects:
        # "regime": regime,
        # "confidence": confidence_score,
        # "leverage": calculated_leverage,
    }

    logger.info(
        f"✅ Signal generated – {signal['direction']} {signal['symbol']} @ {signal['price']}"
    )
    return signal
