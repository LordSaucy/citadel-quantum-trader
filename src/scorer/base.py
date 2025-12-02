# src/scorer/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseScorer(ABC):
    """All scorers must implement a `score(features)` method that returns a float."""
    @abstractmethod
    def score(self, features: Dict[str, float]) -> float:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
