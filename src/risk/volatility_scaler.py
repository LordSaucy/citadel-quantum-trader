# src/risk/volatility_scaler.py
VOL_SCALE = {
    "forex": 1.0,
    "metal": 0.9,
    "index": 0.6,
    "stock": 1.2,
    "future": 0.8,
    "crypto": 3.0,
}


def scale_risk(symbol: str, base_risk_pct: float) -> float:
    asset = classify_asset(symbol)          # helper that maps symbol → class
    return base_risk_pct * VOL_SCALE[asset]
