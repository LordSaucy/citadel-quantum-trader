"""
strategy.py

Signal generation engine for Citadel Quantum Trader.

Provides the `build_signal_function()` factory that returns a strategy callable
compatible with BacktestValidator. The strategy receives a DataFrame of historical
bars (up to and including the current bar) and returns a signal dict if an entry
condition is met.
"""

from typing import Callable, Dict, Optional
import pandas as pd


def build_signal_function(params: Optional[Dict] = None) -> Callable[[pd.DataFrame], Optional[Dict]]:
    """
    Factory function that builds and returns a signal generation callable.
    
    ✅ FIXED: Made 'params' optional and documented its purpose
    
    Args:
        params: Optional dictionary of strategy parameters
                (e.g., weights, thresholds, confluence scores).
                If None, default parameters are used.
    
    Returns:
        A callable that takes a DataFrame of historical bars and returns:
        - A signal dict with keys: symbol, direction, entry_price,
          stop_loss, take_profit, quality_score (optional), confluence_score (optional)
        - None if no entry condition is met
    
    Example:
        >>> signal_func = build_signal_function(params={"threshold": 0.75})
        >>> df = pd.DataFrame({...})  # OHLCV data
        >>> result = signal_func(df)
        >>> if result:
        ...     print(f"BUY {result['symbol']} at {result['entry_price']}")
    """
    # ✅ FIXED: Use default params if none provided
    if params is None:
        params = {}
    
    # =========================================================================
    # ✅ FIXED: Inner signal() function now uses the 'data' parameter
    #           and returns None if no signal (instead of implicit None)
    # =========================================================================
    def signal(data: pd.DataFrame) -> Optional[Dict]:
        """
        Generate a trading signal from historical bar data.
        
        Args:
            data: DataFrame with columns: time, open, high, low, close, volume
                  Rows represent consecutive bars up to and including current.
        
        Returns:
            Signal dict if entry condition is met, None otherwise.
            Signal dict keys:
                - symbol: str (e.g., "EURUSD")
                - direction: str ("BUY" or "SELL")
                - entry_price: float (entry level)
                - stop_loss: float (loss limit)
                - take_profit: float (profit target)
                - quality_score: float (0-100, optional)
                - confluence_score: float (0-100, optional)
        """
        # --------- Sanity checks ---------
        if data is None or data.empty:
            return None
        
        if len(data) < 2:
            # Need at least 2 bars for any meaningful analysis
            return None
        
        # --------- Extract current and previous bars ---------
        current_bar = data.iloc[-1]
        previous_bar = data.iloc[-2]
        
        # --------- Example strategy logic ---------
        # This is a minimal example; replace with your actual strategy
        
        # 1️⃣ Extract OHLCV from current bar
        close = current_bar["close"]
        high = current_bar["high"]
        low = current_bar["low"]
        volume = current_bar["volume"]
        
        # 2️⃣ Calculate a simple indicator (e.g., momentum or confluence score)
        #    For demonstration: compare close to SMA of last 5 bars
        if len(data) >= 5:
            sma_5 = data["close"].tail(5).mean()
            momentum = close - sma_5
        else:
            momentum = 0.0
        
        # 3️⃣ Define entry conditions
        #    Example: BUY if close > SMA and volume > threshold
        threshold = params.get("momentum_threshold", 0.0005)
        min_volume = params.get("min_volume", 1000)
        
        # --------- BUY Signal ---------
        if momentum > threshold and volume > min_volume:
            # Entry: current close
            entry_price = close
            
            # Stop-loss: 1% below entry (configurable via params)
            stop_distance_pct = params.get("stop_loss_pct", 0.01)
            stop_loss = entry_price * (1 - stop_distance_pct)
            
            # Take-profit: 2% above entry (configurable via params)
            tp_distance_pct = params.get("take_profit_pct", 0.02)
            take_profit = entry_price * (1 + tp_distance_pct)
            
            # Confidence score (based on momentum magnitude)
            quality_score = min(100.0, abs(momentum) * 10000)
            
            return {
                "symbol": params.get("symbol", "EURUSD"),
                "direction": "BUY",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "quality_score": quality_score,
                "confluence_score": params.get("confluence_score", 50.0),
            }
        
        # --------- SELL Signal ---------
        if momentum < -threshold and volume > min_volume:
            entry_price = close
            stop_distance_pct = params.get("stop_loss_pct", 0.01)
            stop_loss = entry_price * (1 + stop_distance_pct)
            
            tp_distance_pct = params.get("take_profit_pct", 0.02)
            take_profit = entry_price * (1 - tp_distance_pct)
            
            quality_score = min(100.0, abs(momentum) * 10000)
            
            return {
                "symbol": params.get("symbol", "EURUSD"),
                "direction": "SELL",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "quality_score": quality_score,
                "confluence_score": params.get("confluence_score", 50.0),
            }
        
        # ✅ FIXED: Explicit return None when no signal
        return None
    
    # Return the inner function
    return signal


# =========================================================================
# Example usage
# =========================================================================
if __name__ == "__main__":
    # Create a signal function with custom parameters
    params = {
        "symbol": "EURUSD",
        "momentum_threshold": 0.0005,
        "min_volume": 1500,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.030,
    }
    
    signal_func = build_signal_function(params)
    
    # Simulate historical data
    import numpy as np
    
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    prices = np.cumsum(np.random.randn(100) * 0.0001) + 1.0800
    
    df = pd.DataFrame({
        "time": dates,
        "open": prices,
        "high": prices + 0.0001,
        "low": prices - 0.0001,
        "close": prices,
        "volume": np.random.randint(1000, 5000, 100),
    })
    
    # Generate signals for the entire history
    signals = []
    for i in range(1, len(df)):
        hist_slice = df.iloc[: i + 1]
        sig = signal_func(hist_slice)
        if sig:
            signals.append({
                "time": df.iloc[i]["time"],
                "direction": sig["direction"],
                "entry_price": sig["entry_price"],
            })
    
    print(f"Generated {len(signals)} signals")
    for sig in signals[:5]:
        print(f"  {sig['time']}: {sig['direction']} @ {sig['entry_price']:.5f}")
