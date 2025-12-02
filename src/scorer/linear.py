from .base import BaseScorer
import numpy as np

class LinearScorer(BaseScorer):
    def __init__(self, weights: Dict[str, float]):
        """
        `weights` is a dict mapping lever name → coefficient.
        Example: {"ema_cross": 0.12, "vol_break": 0.08, ...}
        """
        self.weights = weights

    def score(self, features: Dict[str, float]) -> float:
        # Align keys, missing features are treated as 0
        vec = np.array([features.get(k, 0.0) for k in self.weights.keys()], dtype=float)
        w   = np.array(list(self.weights.values()), dtype=float)
        return float(np.dot(w, vec))

    def name(self) -> str:
        return "linear"
