# src/liquidity_risk_scaler.py
def lira_adjusted_risk(base_risk: float, lir: float) -> float:
    """
    base_risk – the risk fraction from the schedule (e.g., 0.005)
    lir       – current Liquidity Imbalance Ratio (0‑1)
    Returns a reduced risk fraction.
    """
    # When LIR = 0.6 → risk is cut by 60 % (effective_risk = base * (1‑0.6))
    # Clamp to a minimum of 10 % of the base risk so we never go to zero.
    adj = max(0.1, 1.0 - lir)   # 0.1 = 10 % of base risk
    return base_risk * adj
