# ultimate_confluence_system/__init__.py
"""
ultimate_confluence_system
~~~~~~~~~~~~~~~~~~~~~~~~~~

High‑level confluence engine for the Citadel Quantum Trader (CQT).

The package bundles all the sub‑modules that compute the various
technical‑analysis signals (Multi‑Timeframe Structure, AOI, Candlestick,
Smart‑Money Concepts, Head‑and‑Shoulders) and aggregates them into a
single “ultimate” score.

Typical usage
-------------

>>> from ultimate_confluence_system import ultimate_system
>>> result = ultimate_system.analyze_complete(
...     symbol="EURUSD",
...     direction="BUY",
...     entry_price=1.0800,
... )
>>> print(result["should_trade"], result["grade"])

The module also re‑exports the core class for advanced use‑cases:

>>> from ultimate_confluence_system import UltimateConfluenceSystem
>>> custom_engine = UltimateConfluenceSystem()
>>> ...

All heavy lifting (logging, weighting, decision logic) lives in
`ultimate_confluence_system.py`.  Keeping the public surface small makes
future refactoring easier and helps static‑analysis tools.
"""

# Re‑export the main class and the ready‑to‑use singleton.
# Importing here ensures that ``import ultimate_confluence_system`` loads the
# implementation only once and that the singleton is instantiated at import
# time (exactly as the original design intended).

from .ultimate_confluence_system import (
    UltimateConfluenceSystem,
    ultimate_system,
)

__all__ = [
    "UltimateConfluenceSystem",
    "ultimate_system",
]
