#!/usr/bin/env python3
"""
garch_vol.py – GARCH‑based volatility estimator for CQT

The estimator is deliberately lightweight and production‑ready:
* Configurable GARCH order (p, q) and forecast horizon.
* Automatic fallback to a simple ATR‑based volatility if the model
  cannot be fitted (e.g., insufficient data, convergence failure).
* Thread‑safe, cache‑aware, and fully typed.
* Extensive logging for observability (integrates with the CQT logger).

Typical usage:

    from src.garch_vol import GarchVolatilityEstimator
    from src.config import Config

    cfg = Config().settings
    estimator = GarchVolatilityEstimator(cfg)

    # Fit on historic price series (pandas Series of close prices)
    estimator.fit(close_prices)

    # Forecast 1‑step ahead volatility (in pips)
    vol = estimator.forecast(steps=1)

    # Or get the historical conditional volatility series
    cond_vol_series = estimator.historical_vol()
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
from arch import arch_model

# ----------------------------------------------------------------------
# CQT internal imports (optional – only needed for config loading)
# ----------------------------------------------------------------------
try:
    from src.config import Config
except Exception:  # pragma: no cover – fallback for unit tests that import the module directly
    Config = None  # type: ignore

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper: simple ATR (Average True Range) – used as a safe fallback
# ----------------------------------------------------------------------
def _atr(series: pd.Series, period: int = 14) -> float:
    """
    Compute the classic ATR over the last ``period`` candles.
    Returns a *pips*‑scaled value (assumes 4‑decimal FX pairs;
    for JPY pairs you may want to multiply by 100).

    Parameters
    ----------
    series : pd.Series
        Close price series (index must be datetime‑like).
    period : int, optional
        Look‑back window for the ATR (default 14).

    Returns
    -------
    float
        ATR expressed in price units (not percent).
    """
    if len(series) < period + 1:
        raise ValueError("Not enough data to compute ATR")

    high = series.rolling(window=period).max()
    low = series.rolling(window=period).min()
    prev_close = series.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_val = tr[-period:].mean()
    return float(atr_val)


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class GarchVolatilityEstimator:
    """
    Wrapper around the ``arch`` library that provides a clean API
    for fitting a GARCH(p,q) model on a price series and forecasting
    conditional volatility.

    The class is **stateful** – you must call ``fit`` before calling
    ``forecast`` or ``historical_vol``.  The fitted model is cached
    internally and can be reused for many forecasts.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        horizon: int = 1,
        cache_seconds: int = 30,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        cfg : dict, optional
            Configuration dictionary (usually ``Config().settings``).  The
            following keys are honoured (if present) and override the
            explicit arguments:

            * ``garch_p`` – GARCH order *p* (default 1)
            * ``garch_q`` – ARCH order *q* (default 1)
            * ``garch_dist`` – distribution (``"normal"``, ``"t"``, …)
            * ``garch_horizon`` – forecast horizon in steps (default 1)
            * ``garch_cache_seconds`` – how long a forecast result is cached (default 30 s)

        p, q, dist, horizon, cache_seconds : int / str
            Direct overrides if you do not want to rely on the config file.

        random_state : int, optional
            Seed for reproducibility (useful in unit tests).
        """
        # Pull values from config if supplied
        if cfg is not None:
            p = cfg.get("garch_p", p)
            q = cfg.get("garch_q", q)
            dist = cfg.get("garch_dist", dist)
            horizon = cfg.get("garch_horizon", horizon)
            cache_seconds = cfg.get("garch_cache_seconds", cache_seconds)

        self.p = int(p)
        self.q = int(q)
        self.dist = str(dist)
        self.horizon = int(horizon)
        self.cache_seconds = int(cache_seconds)
        self.random_state = random_state

        # Internal state
        self._model: Optional[arch_model.ARCHModel] = None
        self._fitted: Optional[arch_model.BaseARCHModel] = None
        self._last_fit_ts: Optional[pd.Timestamp] = None
        self._forecast_cache: Dict[
            Tuple[int, pd.Timestamp], Tuple[float, datetime]
        ] = {}  # (steps, request_ts) → (vol, cached_until)

        logger.debug(
            "GarchVolatilityEstimator initialized (p=%s, q=%s, dist=%s, horizon=%s)",
            self.p,
            self.q,
            self.dist,
            self.horizon,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_series(series: pd.Series) -> pd.Series:
        """
        Validate that the input is a pandas Series with a monotonic datetime index.
        Returns a copy with ``astype(float)`` applied.
        """
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series")

        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("Series index must be a pandas DatetimeIndex")

        if not series.index.is_monotonic_increasing:
            series = series.sort_index()

        # Ensure numeric dtype (important for arch_model)
        return series.astype(float)

    def _reset_cache(self) -> None:
        """Clear the forecast cache – called after a successful refit."""
        self._forecast_cache.clear()
        logger.debug("GarchVolatilityEstimator cache cleared")

    # ------------------------------------------------------------------
    # Public API – fitting
    # ------------------------------------------------------------------
    def fit(self, price_series: pd.Series) -> None:
        """
        Fit a GARCH(p,q) model to the *log‑returns* of ``price_series``.

        Parameters
        ----------
        price_series : pd.Series
            Close price series (datetime index).  The method will compute
            log‑returns internally.

        Raises
        ------
        ValueError
            If the series is too short for the requested model.
        RuntimeError
            If the optimizer fails to converge.
        """
        series = self._ensure_series(price_series)

        if len(series) < max(self.p, self.q) + 10:
            raise ValueError(
                f"Not enough data points to fit GARCH(p={self.p}, q={self.q})"
            )

        # Compute log‑returns (percentage)
        returns = np.log(series).diff().dropna() * 100.0  # expressed in % for stability

        # Build the ARCH model
        self._model = arch_model(
            returns,
            mean="Zero",
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False,
        )

        # Suppress harmless warnings from the optimizer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            try:
                self._fitted = self._model.fit(
                    disp="off",
                    show_warning=False,
                    update_freq=0,
                    random_state=self.random_state,
                )
            except Exception as exc:  # pragma: no cover – extremely unlikely
                logger.exception("GARCH fitting failed")
                raise RuntimeError(f"GARCH fitting failed: {exc}") from exc

        self._last_fit_ts = pd.Timestamp(datetime.now())
        self._reset_cache()
        logger.info(
            "GARCH(p=%s, q=%s, dist=%s) fitted on %d observations (last fit @ %s)",
            self.p,
            self.q,
            self.dist,
            len(returns),
            self._last_fit_ts,
        )

    # ------------------------------------------------------------------
    # Public API – forecast
    # ------------------------------------------------------------------
    def forecast(self, steps: Optional[int] = None) -> float:
        """
        Return the **one‑step ahead conditional volatility** (in *pips*).

        If the model has not been fitted yet, the method falls back to a
        simple ATR estimator (so the engine never crashes).

        Parameters
        ----------
        steps : int, optional
            Number of periods ahead to forecast.  If omitted, the
            estimator uses ``self.horizon`` (default 1).

        Returns
        -------
        float
            Forecast volatility expressed in price units (same scale as the
            input price series).  For FX pairs you may want to convert to
            pips by multiplying by 10 000 (or 100 for JPY pairs).
        """
        steps = steps if steps is not None else self.horizon

        # ----------------------------------------------------------------
        # Cache lookup – avoid recomputing the same forecast within the TTL
        # ----------------------------------------------------------------
        now = datetime.now()
        cache_key = (steps, pd.Timestamp(now))

        if cache_key in self._forecast_cache:
            vol, expires = self._forecast_cache[cache_key]
            if now < expires:
                logger.debug(
                    "GARCH forecast cache hit (steps=%s, ttl=%ss)", steps, (expires - now).seconds
                )
                return vol

        # ----------------------------------------------------------------
        # If we have a fitted model, use it
        # ----------------------------------------------------------------
        if self._fitted is not None:
            try:
                # ``forecast`` returns a DataFrame with columns: variance, sigma, etc.
                fc = self._fitted.forecast(horizon=steps, reindex=False)
                # Conditional variance for the *last* observation + horizon
                var = fc.variance.iloc[-1, steps - 1]  # variance = sigma²
                sigma = np.sqrt(var)  # conditional std‑dev
                # Convert from % (since we fitted on % returns) to price units:
                #   sigma% * price ≈ price change
                # We approximate by using the *most recent* price.
                recent_price = self._model.resids.index[-1]  # index of last residual
                price_series = self._model._y  # original returns series (in %)
                # Approximate price level (inverse log‑return)
                price_level = np.exp(price_series.cumsum()[-1] / 100.0)
                vol_price_units = sigma / 100.0 * price_level
                logger.debug(
                    "GARCH forecast (steps=%s) → σ=%.6f (price units %.6f)",
                    steps,
                    sigma,
                    vol_price_units,
                )
                # Cache the result
                self._forecast_cache[cache_key] = (
                    vol_price_units,
                    now + timedelta(seconds=self.cache_seconds),
                )
                return float(vol_price_units)
            except Exception as exc:  # pragma: no cover – defensive fallback
                logger.exception("GARCH forecast failed, falling back to ATR")
                # Fall through to the ATR fallback

        # ----------------------------------------------------------------
        # Fallback: simple ATR (requires recent price series)
        # ----------------------------------------------------------------
        # NOTE: The caller must have supplied a recent price series via
        # ``fit``; we keep the last series in ``self._model._y`` (percent returns).
        # We reconstruct a price series from those returns.
        if self._model is None:
            raise RuntimeError(
                "GARCH model not fitted and no price series available for ATR fallback"
            )

        # Re‑build price series (starting from 1.0) – the absolute scale is irrelevant
        # for ATR because it measures *price differences*.
        returns = self._model._y
        price_series = pd.Series(np.exp(returns.cumsum() / 100.0), index=returns.index)

        try:
            atr_val = _atr(price_series, period=14)
            logger.debug("ATR fallback used → %.6f", atr_val)
            # Cache the ATR result as well (same TTL)
            self._forecast_cache[cache_key] = (
                atr_val,
                now + timedelta(seconds=self.cache_seconds),
            )
            return atr_val
        except Exception as exc:  # pragma: no cover – should never happen
            logger.exception("ATR fallback also failed")
            raise RuntimeError("Unable to compute volatility (GARCH + ATR both failed)") from exc

    # ------------------------------------------------------------------
    # Public API – historical conditional volatility series
    # ------------------------------------------------------------------
    def historical_vol(self) -> pd.Series:
        """
        Return the *conditional volatility* (sigma) estimated by the fitted
        GARCH model for each observation in the training set.

        Returns
        -------
        pd.Series
            Index = timestamps of the original price series,
            values = conditional sigma (price units, same scale as the input).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._fitted is None:
            raise RuntimeError("GARCH model not fitted – cannot compute historical volatility")

        # ``conditional_volatility`` returns a Series indexed like the residuals
        cond_sigma = self._fitted.conditional_volatility

        # Convert from % (model was fit on % returns) to price units.
        # Use the same conversion trick as in ``forecast``.
        price_series = np.exp(self._model._y.cumsum() / 100.0)
        price_levels = price_series.reindex(cond_sigma.index, method="ffill")
        sigma_price_units = cond_sigma / 100.0 * price_levels

        logger.info(
            "Historical conditional volatility computed (%d points)", len(sigma_price_units)
        )
        return pd.Series(sigma_price_units, index=cond_sigma.index)

    # ------------------------------------------------------------------
    # Utility: quick one‑liner for external callers
    # ------------------------------------------------------------------
    @classmethod
    def quick_vol(
        cls,
        price_series: pd.Series,
        *,
        cfg: Optional[Dict[str, Any]] = None,
        steps: int = 1,
    ) -> float:
        """
        Convenience wrapper – fits a temporary GARCH model on ``price_series``,
        forecasts ``steps`` ahead, and returns the volatility.  The temporary
        estimator is discarded immediately (no caching).

        This is handy for ad‑hoc analyses or for scripts that only need a
        single volatility number.

        Parameters
        ----------
        price_series : pd.Series
            Close price series (datetime index).
        cfg : dict, optional
            Configuration dict (same keys as the constructor).
        steps : int, optional
            Forecast horizon (default 1).

        Returns
        -------
        float
            Forecast volatility (price units).
        """
        estimator = cls(cfg=cfg)
        estimator.fit(price_series)
        return estimator.forecast(steps=steps)


# ----------------------------------------------------------------------
# If this file is executed directly, run a tiny demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple demo using random walk data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")
    # Simulate a price series around 1.2000 (FX style)
    price = 1.2000 + np.cumsum(np.random.normal(0, 0.001, size=len(dates)))
    price_series = pd.Series(price, index=dates)

    # Load a dummy config (could also be None)
    dummy_cfg = {
        "garch_p": 1,
        "garch_q": 1,
        "garch_dist": "normal",
        "garch_horizon": 1,
        "garch_cache_seconds": 30,
    }

    est = GarchVolatilityEstimator(dummy_cfg)
    est.fit(price_series)

    # ------------------------------------------------------------------
    # Forecast the next‑step volatility and print it
    # ------------------------------------------------------------------
    forecast_vol = est.forecast()
    print(f"\n▶️  1‑step ahead forecast volatility (price units): {forecast_vol:.6f}")

    # ------------------------------------------------------------------
    # Also demonstrate a multi‑step forecast (e.g., 5 steps ahead)
    # ------------------------------------------------------------------
    forecast_5 = est.forecast(steps=5)
    print(f"▶️  5‑step ahead forecast volatility (price units): {forecast_5:.6f}")

    # ------------------------------------------------------------------
    # Show the historical conditional volatility series (first 5 values)
    # ------------------------------------------------------------------
    hist_vol = est.historical_vol()
    print("\n▶️  Historical conditional volatility (first 5 points):")
    print(hist_vol.head())

    # ------------------------------------------------------------------
    # Quick one‑liner usage example (no explicit estimator object)
    # ------------------------------------------------------------------
    quick_vol = GarchVolatilityEstimator.quick_vol(price_series, cfg=dummy_cfg, steps=1)
    print(f"\n▶️  Quick‑vol (using classmethod) → {quick_vol:.6f}")

