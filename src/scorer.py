# src/scorer.py
import json
import pathlib
import numpy as np
import lightgbm as lgb
import joblib
from typing import Dict, Callable

# -------------------------------------------------
# 1️⃣ Linear scorer (kept for backward compatibility)
# -------------------------------------------------
class LinearScorer:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def score(self, features: Dict[str, float]) -> float:
        # dot‑product between feature dict and weight dict
        return sum(self.weights.get(k, 0.0) * v for k, v in features.items())


# -------------------------------------------------
# 2️⃣ LightGBM scorer
# -------------------------------------------------
class LightGBMScorer:
    """
    Wrapper around a trained LightGBM Booster.
    The model expects the same feature order that was used during training.
    """

    def __init__(self, model_path: pathlib.Path):
        # Load the native binary – fastest
        self.booster = lgb.Booster(model_file=str(model_path))

        # Store the feature order (LightGBM saves it in the model)
        self.feature_names = self.booster.feature_name()

    def score(self, features: Dict[str, float]) -> float:
        """
        LightGBM returns a probability (0‑1) for the positive class.
        We treat that as the “raw signal strength”.  If you need a
        calibrated R‑multiple you can map the probability to a
        custom scale here.
        """
        # Build a 2‑D numpy array in the exact order LightGBM expects
        arr = np.array([[features.get(name, 0.0) for name in self.feature_names]],
                       dtype=np.float32)
        prob = self.booster.predict(arr)[0]   # returns a scalar
        return float(prob)


# -------------------------------------------------
# 3️⃣ Factory – decide which scorer to instantiate
# -------------------------------------------------
def get_scorer(mode: str = "linear") -> Callable[[Dict[str, float]], float]:
    """
    mode:
        "linear"  – use the simple weighted sum (legacy)
        "lightgbm" – load the trained LightGBM model
    Returns a callable `score(features_dict) -> float`.
    """
    if mode == "lightgbm":
        # Path is relative to the repo root; adjust if you store elsewhere
        model_path = pathlib.Path("models/lightgbm_model.txt")
        if not model_path.is_file():
            raise FileNotFoundError(f"LightGBM model not found at {model_path}")
        scorer = LightGBMScorer(model_path)
        return scorer.score

    # Default – linear
    # Load the weight vector from a JSON file (you already have one)
    weights_path = pathlib.Path("config/lever_weights.json")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file missing: {weights_path}")
    with weights_path.open() as f:
        weights = json.load(f)
    scorer = LinearScorer(weights)
    return scorer.score
