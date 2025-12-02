import json, numpy as np
from pathlib import Path
import pandas as pd
import json
import numpy as np
from prometheus_client import Gauge
import logging


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
       
log = logging.getLogger(__name__)

BASELINE_PATH = Path("/app/drift/baseline.json")
# Prometheus gauges – one per feature (dynamic registration)
psi_gauges = {}          # type: dict[str, Gauge]
kl_gauges  = {}          # optional, if you also compute KL

# -----------------------------------------------------------------
# Helper: load baseline (edges + density) for each feature
# -----------------------------------------------------------------
def load_baseline() -> dict:
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError("Baseline drift file not found")
    return json.loads(BASELINE_PATH.read_text())

# -----------------------------------------------------------------
# Compute PSI for a single feature
# -----------------------------------------------------------------
def _psi(expected_density, actual_density, eps=1e-6):
    expected = np.where(expected_density == 0, eps, expected_density)
    actual   = np.where(actual_density   == 0, eps, actual_density)
    return np.sum((expected - actual) * np.log(expected / actual))

# -----------------------------------------------------------------
# Compute KL‑divergence for a single feature (optional)
# -----------------------------------------------------------------
def _kl(expected_density, actual_density, eps=1e-6):
    expected = np.where(expected_density == 0, eps, expected_density)
    actual   = np.where(actual_density   == 0, eps, actual_density)
    return np.sum(actual * np.log(actual / expected))

# -----------------------------------------------------------------
# Main routine – called by FastAPI background task
# -----------------------------------------------------------------
def run_drift_check(df: pd.DataFrame):
    """
    df – new data (same columns as baseline). Typically the most recent
    24‑h or 7‑d window of raw feature values (pre‑processed, same scaling).
    """
    baseline = load_baseline()
    for col in df.columns:
        if col not in baseline:
            log.warning(f"Drift: column {col} missing from baseline – skipping")
            continue

        # ---------------------------------------------------------
        # Bin the new data using the SAME edges as the baseline
        # ---------------------------------------------------------
        edges = np.array(baseline[col]["edges"])
        hist, _ = np.histogram(df[col].dropna(), bins=edges, density=True)

        # ---------------------------------------------------------
        # Compute PSI (and optionally KL)
        # ---------------------------------------------------------
        psi_val = _psi(np.array(baseline[col]["density"]), hist)
        kl_val  = _kl(np.array(baseline[col]["density"]), hist)

        # ---------------------------------------------------------
        # Register / update Prometheus gauges (dynamic)
        # ---------------------------------------------------------
        if col not in psi_gauges:
            psi_gauges[col] = Gauge(
                f"feature_psi_{col}",
                f"Population Stability Index for feature {col}",
            )
        psi_gauges[col].set(psi_val)

        if col not in kl_gauges:
            kl_gauges[col] = Gauge(
                f"feature_kl_{col}",
                f"KL‑divergence for feature {col}",
            )
        kl_gauges[col].set(kl_val)

        # ---------------------------------------------------------
        # Log if drift exceeds a threshold (configurable per‑feature)
        # ---------------------------------------------------------
        if psi_val > 0.25:   # moderate‑to‑severe drift
            log.warning(
                f"⚠️ Drift detected on {col}: PSI={psi_val:.3f} (threshold 0.25)"
            )
        if kl_val > 0.1:      # arbitrary KL threshold
            log.warning(
                f"⚠️ KL drift on {col}: KL={kl_val:.3f}"
            )
