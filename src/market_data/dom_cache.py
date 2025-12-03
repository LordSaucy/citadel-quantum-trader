# src/market_data/dom_cache.py
#!/usr/bin/env python3
import threading
import pandas as pd
from .ibkr_dom.get_dom import get_dom
from .ibkr_dom.connect_ibkr import connect_ibkr

class DomCache:
    """
    Maintains a live, thread‑safe copy of the DOM for a set of symbols.
    The cache is refreshed in a background thread that receives incremental
    updates from IBKR (via ib_insync callbacks).  For simplicity we poll
    every 0.5 s – you can replace the polling loop with the ib_insync
    `updateMktDepth` callback if you need ultra‑low latency.
    """
    def __init__(self, symbols: list[str], depth: int = 20, poll_interval: float = 0.5):
        self._symbols = symbols
        self._depth   = depth
        self._interval = poll_interval
        self._lock    = threading.RLock()
        self._data: dict[str, pd.DataFrame] = {}   # symbol → DataFrame
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self):
        ib = connect_ibkr()
        while not self._stop_event.is_set():
            for sym in self._symbols:
                try:
                    df = get_dom(ib, sym, depth=self._depth)
                    with self._lock:
                        self._data[sym] = df
                except Exception as exc:
                    # Log but continue – a temporary hiccup shouldn't kill the thread
                    import logging
                    logging.getLogger(__name__).warning("DOM fetch failed for %s: %s", sym, exc)
            self._stop_event.wait(self._interval)

    def get_latest(self, symbol: str) -> pd.DataFrame | None:
        """Return the most recent DOM DataFrame for `symbol` (or None)."""
        with self._lock:
            return self._data.get(symbol).copy() if symbol in self._data else None
