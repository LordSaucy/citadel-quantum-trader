#!/usr/bin/env python3
"""
hmm_regime.py – Hidden‑Markov‑Model based market‑regime classifier

The module provides a single public class ``HMMRegime`` that can be
instantiated once at process start and used throughout the trading
engine to label each incoming price bar with a discrete regime:

    0 – “Low‑volatility / ranging”
    1 – “Medium‑volatility / trending”
    2 – “High‑volatility / breakout”

The classifier is trained offline on historical OHLCV data (the
training routine is deliberately separated from the inference path so
that the production container only needs to *load* a pre‑trained model).

If the model cannot be loaded (e.g. missing file, version mismatch) the
class automatically falls back to a very simple rule‑based regime
detector, ensuring the rest of the system continues to operate.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------
from src.config import Config

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper types
# ----------------------------------------------------------------------
ArrayLike = Union[np.ndarray, List[float], List[int]]


# ----------------------------------------------------------------------
# HMMRegime – main public class
# ----------------------------------------------------------------------
class HMMRegime:
    """
    Hidden‑Markov‑Model based regime classifier.

    Typical usage pattern::

        cfg = Config().settings
        regime = HMMRegime(cfg)
        regime.load()                     # load a pre‑trained model (if available)
        regime_label = regime.predict(bar)   # bar = dict with OHLCV fields

    The class is **thread‑safe** – a re‑entrant lock protects the internal
    ``GaussianHMM`` instance during ``predict`` and ``fit`` calls.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any] | None = None):
        """
        Parameters
        ----------
        cfg : dict or None
            Configuration dictionary (normally obtained via ``Config().settings``).
            Expected keys (all optional, defaults are shown):

            - ``hmm.n_components`` (int) – number of hidden states (default 3).
            - ``hmm.covariance_type`` (str) – ``"diag"``, ``"full"``, etc. (default ``"diag"``).
            - ``hmm.max_iter`` (int) – EM iterations for training (default 100).
            - ``hmm.model_path`` (str) – filesystem path where the model is persisted.
            - ``hmm.features`` (list) – list of feature column names used for training/prediction.
        """
        self.cfg = cfg or Config().settings

        # --------------------------------------------------------------
        # Model hyper‑parameters (with sensible defaults)
        # --------------------------------------------------------------
        self.n_components: int = int(self.cfg.get("hmm.n_components", 3))
        self.covariance_type: str = self.cfg.get(
            "hmm.covariance_type", "diag"
        )  # diag is fast & works well for finance
        self.max_iter: int = int(self.cfg.get("hmm.max_iter", 100))

        # --------------------------------------------------------------
        # Feature engineering – which columns from the bar we actually use
        # --------------------------------------------------------------
        self.feature_names: List[str] = self.cfg.get(
            "hmm.features",
            [
                "log_return",
                "atr",
                "volume_change",
                "price_range",  # high‑low spread as % of close
            ],
        )

        # --------------------------------------------------------------
        # Persistence location
        # --------------------------------------------------------------
        default_path = Path.cwd() / "models" / "hmm_regime.pkl"
        self.model_path: Path = Path(
            self.cfg.get("hmm.model_path", str(default_path))
        ).expanduser().resolve()

        # --------------------------------------------------------------
        # Internal objects – created lazily
        # --------------------------------------------------------------
        self._model: hmm.GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        self._lock = threading.RLock()

        # --------------------------------------------------------------
        # Fallback rule‑based regime (used when model cannot be loaded)
        # --------------------------------------------------------------
        self._fallback = SimpleRegimeFallback(self.cfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _ensure_model_loaded(self) -> None:
        """Raise a clear error if the model is not available."""
        if self._model is None or self._scaler is None:
            raise RuntimeError(
                "HMMRegime model not loaded – call .load() or .fit() first."
            )

    # ------------------------------------------------------------------
    # Public API – model persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """
        Serialize the fitted HMM and the scaler to ``self.model_path``.
        The file is written atomically (temp file → rename) to avoid
        partially‑written models on crash.
        """
        with self._lock:
            self._ensure_model_loaded()
            payload = {
                "model": self._model,
                "scaler": self._scaler,
                "cfg": {
                    "n_components": self.n_components,
                    "covariance_type": self.covariance_type,
                    "max_iter": self.max_iter,
                    "feature_names": self.feature_names,
                },
            }

            tmp_path = self.model_path.with_suffix(".tmp")
            try:
                # ``hmmlearn`` models are pickle‑compatible
                import joblib

                joblib.dump(payload, tmp_path)
                tmp_path.replace(self.model_path)
                log.info("HMMRegime model saved to %s", self.model_path)
            except Exception as exc:  # pragma: no cover
                log.exception("Failed to save HMMRegime model: %s", exc)
                raise

    def load(self) -> bool:
        """
        Load a previously saved model from ``self.model_path``.
        Returns ``True`` if loading succeeded, ``False`` otherwise (fallback will be used).

        The method is tolerant to version mismatches – if the pickle cannot be
        deserialized it logs the error and returns ``False``.
        """
        if not self.model_path.is_file():
            log.warning(
                "HMMRegime model file not found at %s – using fallback.",
                self.model_path,
            )
            return False

        with self._lock:
            try:
                import joblib

                payload = joblib.load(self.model_path)
                self._model = payload["model"]
                self._scaler = payload["scaler"]
                # sanity‑check that the loaded config matches the current one
                loaded_cfg = payload.get("cfg", {})
                if loaded_cfg.get("n_components") != self.n_components:
                    log.warning(
                        "Loaded HMMRegime model has %s components, but config expects %s. "
                        "Proceeding with loaded model.",
                        loaded_cfg.get("n_components"),
                        self.n_components,
                    )
                log.info("HMMRegime model loaded from %s", self.model_path)
                return True
            except Exception as exc:  # pragma: no cover
                log.exception(
                    "Failed to load HMMRegime model from %s – using fallback.", self.model_path
                )
                self._model = None
                self._scaler = None
                return False

    # ------------------------------------------------------------------
    # Public API – training (offline)
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train a new HMM on the supplied DataFrame and persist it.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain the columns listed in ``self.feature_names``.
            The method will compute a standard‑scaler, fit the GaussianHMM,
            and then call ``self.save()``.
        """
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(
                f"The training DataFrame is missing required columns: {missing}"
            )

        with self._lock:
            # ----------------------------------------------------------
            # 1️⃣  Scale the features (zero‑mean, unit‑variance)
            # ----------------------------------------------------------
            scaler = StandardScaler()
            X = scaler.fit_transform(df[self.feature_names].astype(float).values)

            # ----------------------------------------------------------
            # 2️⃣  Fit the Gaussian HMM
            # ----------------------------------------------------------
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.max_iter,
                verbose=False,
                random_state=42,
            )
            model.fit(X)

            # ----------------------------------------------------------
            # 3️⃣  Store and persist
            # ----------------------------------------------------------
            self._model = model
            self._scaler = scaler
            self.save()
            log.info(
                "HMMRegime training complete – %s hidden states, %s features",
                self.n_components,
                len(self.feature_names),
            )

    # ------------------------------------------------------------------
    # Public API – inference
    # ------------------------------------------------------------------
    def predict(self, bar: Dict[str, Any]) -> int:
        """
        Infer the regime for a single market bar.

        Parameters
        ----------
        bar : dict
            Must contain the keys listed in ``self.feature_names``.
            Values can be numeric or strings that can be cast to ``float``.

        Returns
        -------
        int
            The most likely hidden state (0 … n_components‑1).  If the model
            is unavailable the method delegates to the simple fallback and
            returns its regime label.
        """
        # ----------------------------------------------------------
        # Fast path – if the model is not loaded, use fallback
        # ----------------------------------------------------------
        if self._model is None or self._scaler is None:
            return self._fallback.predict(bar)

        # ----------------------------------------------------------
        # 1️⃣  Extract and scale the feature vector
        # ----------------------------------------------------------
        try:
            raw_vec = np.array([float(bar[name]) for name in self.feature_names])
        except KeyError as exc:
            log.error("Missing feature %s in bar – using fallback", exc.args[0])
            return self._fallback.predict(bar)
        except (TypeError, ValueError) as exc:
            log.error("Non‑numeric feature in bar – using fallback: %s", exc)
            return self._fallback.predict(bar)

        with self._lock:
            # ``reshape(1, -1)`` because HMM expects a 2‑D array (samples × features)
            X_scaled = self._scaler.transform(raw_vec.reshape(1, -1))

            # ------------------------------------------------------
            # 2️⃣  Decode the most probable hidden state sequence
            # ------------------------------------------------------
            hidden_states = self._model.predict(X_scaled)  # shape (1,)
            regime = int(hidden_states[0])

            log.debug(
                "HMMRegime prediction – bar=%s → regime=%s",
                {k: bar[k] for k in self.feature_names},
                regime,
            )
            return regime

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def decode_sequence(self, df: pd.DataFrame) -> List[int]:
        """
        Decode the hidden‑state sequence for an entire DataFrame.
        Useful for back‑testing or visualisation.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain the required feature columns.

        Returns
        -------
        list[int]
            List of regime labels (length = len(df)).
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not loaded – cannot decode sequence.")

        X = self._scaler.transform(df[self.feature_names].astype(float).values)
        with self._lock:
            states = self._model.predict(X)
        return list(map(int, states))


# ----------------------------------------------------------------------
# Simple rule‑based fallback (used when the HMM cannot be loaded)
# ----------------------------------------------------------------------
class SimpleRegimeFallback:
    """
    Very lightweight regime detector that mimics the output shape of the
    HMM.  It uses a few handcrafted heuristics based on volatility
    and price range.

    The implementation is deliberately *deterministic* and has **no
    external dependencies**, making it safe for production even when the
    HMM model file is corrupted or missing.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or Config().settings
        # Thresholds can be overridden via config.yaml
        self.low_vol_thresh = float(self.cfg.get("fallback.low_vol_thresh", 0.5))
        self.high_vol_thresh = float(self.cfg.get("fallback.high_vol_thresh", 2.0))

    def predict(self, bar: Dict[str, Any]) -> int:
        """
        Heuristic regime classification:

        * **0 – Low volatility**   : ``abs(log_return) < low_vol_thresh``
        * **1 – Medium volatility**: ``low_vol_thresh ≤ |log_return| < high_vol_thresh``
        * **2 – High volatility**  : ``|log_return| ≥ high_vol_thresh``

        The function expects the bar to contain a pre‑computed
        ``log_return`` field (``np.log(close / prev_close)``).  If the field
        is missing we fall back to a neutral ``0`` regime.
        """
        try:
            lr = float(bar["log_return"])
        except (KeyError, TypeError, ValueError):
            log.debug(
                "SimpleRegimeFallback: missing or invalid log_return – defaulting to regime 0"
            )
            return 0

        abs_lr = abs(lr)
        if abs_lr < self.low_vol_thresh:
            return 0
        if abs_lr < self.high_vol_thresh:
            return 1
        return 2
