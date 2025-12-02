import numpy as np
import json
from pathlib import Path

BASELINE_PATH = Path("/app/drift/baseline.json")

def build_baseline(df: pd.DataFrame, n_bins: int = 50) -> dict:
    baseline = {}
    for col in df.columns:
        hist, edges = np.histogram(df[col].dropna(), bins=n_bins, density=True)
        baseline[col] = {"edges": edges.tolist(), "density": hist.tolist()}
    BASELINE_PATH.write_text(json.dumps(baseline))
    return baseline
