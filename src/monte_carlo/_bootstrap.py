# src/monte_carlo/_bootstrap.py
"""
Monte‑Carlo / bootstrap utilities.

Given a list of realised trade P&L values (or any numeric series),
draw *iterations* random samples **with replacement** and compute
summary statistics for each draw.  The final output aggregates those
statistics to produce confidence‑interval estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict, List


def _draw_sample(pnl_series: np.ndarray, size: int) -> np.ndarray:
    """
    Return a single bootstrap sample of length ``size`` drawn **with**
    replacement from ``pnl_series``.
    """
    return np.random.choice(pnl_series, size=size, replace=True)


def _sample_statistics(sample: np.ndarray) -> Dict[str, float]:
    """
    Compute the statistics we care about for a *single* bootstrap sample.

    Returns a dict with the same keys that will be aggregated later.
    """
    total = sample.sum()
    mean = sample.mean()
    median = np.median(sample)
    std = sample.std(ddof=1)

    # Value‑at‑Risk (5 % tail) and Conditional VaR (expected loss beyond VaR)
    var_5 = np.percentile(sample, 5)          # 5 % worst loss
    cvar_5 = sample[sample <= var_5].mean()   # average of the worst 5 %

    # Simple win‑rate / expectancy on the *sample*
    wins = (sample > 0).sum()
    win_rate = wins / len(sample) if len(sample) else 0.0
    expectancy = total / len(sample) if len(sample) else 0.0

    # Profit factor (gross profit / gross loss)
    gross_profit = sample[sample > 0].sum()
    gross_loss = -sample[sample < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss else np.inf

    return {
        "total": float(total),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "var_5": float(var_5),
        "cvar_5": float(cvar_5),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
    }


def run_bootstrap(
    pnl_series: Iterable[float],
    iterations: int = 10_000,
    random_seed: int | None = None,
) -> List[Dict[str, float]]:
    """
    Perform ``iterations`` bootstrap draws on ``pnl_series`` and return a
    list of per‑draw statistic dictionaries.

    Parameters
    ----------
    pnl_series : iterable of float
        The realised per‑trade P&L values (net of commissions, spread,
        slippage – i.e. the *net* profit you already have in the back‑test CSV).
    iterations : int, default 10 000
        Number of bootstrap samples.
    random_seed : int | None
        For reproducibility when you need it (e.g. in CI).

    Returns
    -------
    List[Dict[str, float]]
        One dict per iteration, each containing the keys produced by
        ``_sample_statistics``.
    """
    pnl_arr = np.asarray(list(pnl_series), dtype=float)

    if random_seed is not None:
        np.random.seed(random_seed)

    results: List[Dict[str, float]] = []
    n = len(pnl_arr)

    for _ in range(iterations):
        sample = _draw_sample(pnl_arr, size=n)
        results.append(_sample_statistics(sample))

    return results
