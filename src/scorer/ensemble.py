# src/scorer/ensemble.py
from .base import BaseScorer
from typing import List, Tuple

class WeightedEnsemble(BaseScorer):
    """
    Takes a list of (scorer_instance, weight) tuples.
    The final score = Σ weight_i * scorer_i.score(features)
    Weights are normalised to sum to 1.
    """
    def __init__(self, components: List[Tuple[BaseScorer, float]]):
        total = sum(w for _, w in components)
        if total == 0:
            raise ValueError("Ensemble weights must sum > 0")
        self.components = [(sc, w / total) for sc, w in components]

    def score(self, features: dict) -> float:
        return sum(weight * scorer.score(features)
                   for scorer, weight in self.components)

    def name(self) -> str:
        names = "+".join([sc.name() for sc, _ in self.components])
        return f"ensemble({names})"
