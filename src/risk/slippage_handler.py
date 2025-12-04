# src/risk/slippage_handler.py
SLIPPAGE_MAX = {
    "forex": 2,          # pips
    "metal": 0.5,        # ounces
    "index": 1,          # points
    "stock": 0.05,       # dollars
    "future": 0.5,
    "crypto": 0.01,      # USD
}


def max_allowed_slippage(symbol: str) -> float:
    asset = classify_asset(symbol)
    return SLIPPAGE_MAX[asset]
