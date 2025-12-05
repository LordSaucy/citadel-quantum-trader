# src/optimizer/__init__.py
from .master_optimizer import MasterWinRateOptimizer, OverallResult

# Global singleton – importable from anywhere (e.g. src/main.py)
master_optimizer = MasterWinRateOptimizer()
