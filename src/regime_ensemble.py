import json
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances_argmin_min

BANK_PATH = Path("/app/artifacts/current_regime_bank.json")
THRESHOLD_SIMILARITY = 0.25   # tune – lower = stricter match

def load_bank():
    with BANK_PATH.open() as f:
        data = json.load(f)
    centroids = np.array([c["centroid"] for c in data["clusters"]])
    meta = {c["id"]: c for c in data["clusters"]}
    return centroids, meta

CENTROIDS, META = load_bank()

def current_regime_vector(lstm_probs, hmm_state, garch_vol):
    """
    Build the same feature vector that was used for clustering.
    lstm_probs: np.ndarray shape (3,) – [exp, neut, contr]
    hmm_state: int (0/1/2)
    garch_vol: float (forecasted variance)
    """
    # one‑hot encode HMM state
    hmm_onehot = np.zeros(3)
    hmm_onehot[hmm_state] = 1.0

    # normalize garch_vol (use the same scaler you used offline;
    # for simplicity we just z‑score using the training mean/std stored in the bank)
    # Assume the bank JSON also contains "garch_mean" and "garch_std".
    garch_norm = (garch_vol - META[0]["garch_mean"]) / META[0]["garch_std"]

    vec = np.concatenate([lstm_probs, hmm_onehot, [garch_norm]])
    return vec

def match_regime(vec):
    """
    Returns (cluster_id, similarity_score) or (None, 0) if no match.
    Similarity = 1 - normalized Euclidean distance (0–1).
    """
    idx, dist = pairwise_distances_argmin_min([vec], CENTROIDS)
    # Convert distance to similarity (max distance in the space is sqrt(dim))
    max_dist = np.sqrt(vec.shape[0])
    similarity = 1 - (dist[0] / max_dist)

    if similarity >= THRESHOLD_SIMILARITY:
        cluster_id = int(idx[0])
        return cluster_id, similarity
    return None, similarity
