# ---------------------------------------------------------
# signal_generator.py – builds the confluence score
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from features import FEATURES, load_regime_weights
from config_loader import Config   # a tiny wrapper that loads config.yaml
from utils import standardize      # optional z‑score normalizer

cfg = Config().settings

class SignalEngine:
    def __init__(self):
        # Load the three regime‑specific weight vectors once at start‑up.
        self.weights = {
            "trend": load_regime_weights(cfg, "trend"),
            "range": load_regime_weights(cfg, "range"),
            "high_vol": load_regime_weights(cfg, "high_vol")
        }

    def _select_weights(self, regime_label: str) -> pd.Series:
        """Return the weight vector that matches the current regime."""
        return self.weights.get(regime_label, self.weights["trend"])

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply every enabled feature and concatenate the results.
        Returns a DataFrame where each column is a feature series.
        """
        feats = {}
        for name, fn in FEATURES.items():
            # Pull any feature‑specific params from config.yaml
            params = cfg.get('features', {}).get(name, {})
            try:
                feats[name] = fn(df, **params)
            except Exception as exc:
                # Log and fall back to NaN series – the scorer will ignore it.
                print(f"[WARN] Feature {name} failed: {exc}")
                feats[name] = pd.Series([np.nan] * len(df), index=df.index)

        feature_df = pd.DataFrame(feats, index=df.index)
        return feature_df

    def score(self,
              df: pd.DataFrame,
              regime_label: str = "trend",
              normalize: bool = True) -> pd.Series:
        """
        Main entry point used by the bot:
        1️⃣ Compute raw features.
        2️⃣ (Optionally) standardize them.
        3️⃣ Multiply by the regime‑specific weight vector.
        4️⃣ Sum → confluence score.
        """
        raw_feats = self.compute_features(df)

        # Optional normalization – prevents a feature with huge magnitude
        # (e.g., raw ATR) from dominating the dot‑product.
        if normalize:
            feats = raw_feats.apply(standardize)
        else:
            feats = raw_feats

        # Align the weight vector with the feature columns.
        w = self._select_weights(regime_label)
        # If a new feature was added but not present in the weight file,
        # fill its weight with 0 (ignore it) and warn.
        missing = set(feats.columns) - set(w.index)
        if missing:
            print(f"[INFO] Missing weights for {missing}; assigning 0.")
            for m in missing:
                w[m] = 0.0

        # Ensure ordering matches.
        w = w[feats.columns]

        # Dot product → scalar score per row.
        score_series = (feats * w).sum(axis=1)

        return score_series
