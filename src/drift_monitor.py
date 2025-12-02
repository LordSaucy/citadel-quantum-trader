import json, numpy as np
from pathlib import Path
import pandas as pd

BASELINE_PATH = Path("/app/drift/baseline.json")
THRESHOLD_PSI = 0.25   # configurable

def _psi(expected, actual, eps=1e-6):
    """Calculate PSI for a single feature (both arrays are densities)."""
    # Avoid division by zero
    expected = np.where(expected == 0, eps, expected)
    actual = np.where(actual == 0, eps, actual)
    return np.sum((expected - actual) * np.log(expected / actual))

def compute_psi(df: pd.DataFrame) -> dict:
    baseline = json.loads(BASELINE_PATH.read_text())
    psi_scores = {}
    for col in df.columns:
        if col not in baseline:
            continue
        edges = np.array(baseline[col]["edges"])
       
