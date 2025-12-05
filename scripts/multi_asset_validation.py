#!/usr/bin/env python3
"""
Run the full validation suite across a basket of uncorrelated assets.
"""


import itertools
from datetime import datetime, timedelta
from src.data_feed.feed_factory import get_feed
from src.backtest.validator import BacktestValidator
from src.strategy.my_strategy import generate_signal
from src.risk.volatility_scaler import scale_risk


ASSETS = [
    ("EURUSD", "forex"),
    ("XAUUSD", "metal"),
    ("SPX",    "index"),
    ("AAPL",   "stock"),
    ("BTCUSDT","crypto"),
]


def main():
    for symbol, asset_class in ASSETS:
        print(f"\n=== VALIDATING {symbol} ({asset_class}) ===")
        feed = get_feed(asset_class)
        validator = BacktestValidator(
            initial_balance=100_000,
            risk_per_trade=scale_risk(symbol, 1.0)   # 1% baseline
        )
        analysis = validator.run_validation(
            symbol=symbol,
            timeframe=5,
            start_date=datetime.now() - timedelta(days=180),
            end_date=datetime.now(),
            strategy_function=generate_signal,
            data_feed=feed,
            min_win_rate=0.99   # 99% win‑rate threshold
        )
        print(f"Win‑rate: {analysis['win_rate']*100:.2f}%")
        print(f"Draw‑down: {analysis['max_drawdown']:.2f}%")
        print(f"Sharpe: {analysis['sharpe']:.2f}")


if __name__ == "__main__":
    main()
