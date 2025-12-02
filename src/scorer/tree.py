# src/scorer/tree.py
from .base import BaseScorer
import xgboost as xgb
import numpy as np
import json
from typing import Dict

class TreeScorer(BaseScorer):
    def __init__(self, model_path: str, feature_order: list):
        """
        `model_path` – XGBoost binary model (saved with Booster.save_model()).
        `feature_order` – list of feature names in the exact order the model expects.
        """
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)
        self.feature_order = feature_order

    def score(self, features: Dict[str, float]) -> float:
        # Build a DMatrix with a single row
        row = np.array([features.get(k, 0.0) for k in self.feature_order], dtype=float)
        dmatrix = xgb.DMatrix(row.reshape(1, -1))
        # XGBoost returns a list; we take the first element
        pred = self.booster.predict(dmatrix)[0]
        return float(pred)

    def name(self) -> str:
        return "tree"
