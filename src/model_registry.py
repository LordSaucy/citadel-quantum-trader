import yaml
import joblib
import numpy as np
from pathlib import Path

class ModelRegistry:
    def __init__(self, cfg_path: Path = Path("/app/config/config.yaml")):
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        self.weights = cfg.get("model_voting", {"old": 0.2, "medium": 0.5, "fast": 0.3})
        # Load the three models (they can be any sklearn‑compatible estimator)
        self.models = {
            "old": joblib.load("/app/models/old.pkl"),
            "medium": joblib.load("/app/models/medium.pkl"),
            "fast": joblib.load("/app/models/fast.pkl"),
        }

    def _score(self, model, feats: np.ndarray) -> float:
        """Return a probability‑like score (0‑1)."""
        # Most sklearn models expose `predict_proba`; fall back to `predict`
        if hasattr(model, "predict_proba"):
            return model.predict_proba(feats)[0, 1]
        return float(model.predict(feats))

    def vote(self, feature_dict: dict) -> bool:
        """Return True (signal = “win”) if the weighted vote crosses 0.5."""
        # Convert dict → 2‑D numpy array expected by sklearn
        feats = np.array([list(feature_dict.values())])
        weighted_sum = 0.0
        total_weight = 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0)
            if w <= 0:
                continue
            s = self._score(model, feats)
            weighted_sum += w * s
            total_weight += w
        # Normalise to [0,1]
        final_score = weighted_sum / total_weight if total_weight else 0.0
        # Threshold can be a config param; default 0.5
        return final_score >= 0.5
