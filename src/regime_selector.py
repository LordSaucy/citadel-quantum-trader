# src/regime_selector.py
import joblib, json, numpy as np
from pathlib import Path

class RegimeSelector:
    """
    Loads the pre‑trained classifier and returns a regime id (0,1,2).
    The selector is called once per bar before the main scoring step.
    """
    def __init__(self, model_path: Path, feat_path: Path):
        self.clf = joblib.load(model_path)
        self.features = json.load(open(feat_path))

    def predict(self, bar: dict) -> int:
        """
        `bar` – dict with keys matching `self.features`.
        Returns an integer regime id.
        """
        vec = np.array([bar.get(f, 0.0) for f in self.features]).reshape(1, -1)
        return int(self.clf.predict(vec)[0])
