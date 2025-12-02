import numpy as np
import json
from pathlib import Path

class BayesianWeightUpdater:
    """
    Simple Gaussian‑conjugate updater for a linear scorer.
    Stores mu (posterior mean) and sigma2 (posterior variance) per lever.
    """
    def __init__(self, prior_path: Path, obs_noise: float = 0.01):
        """
        prior_path – JSON with {"mu0": {...}, "sigma0": {...}}
        """
        data = json.loads(prior_path.read_text())
        self.mu0    = np.array([data["mu0"][k] for k in sorted(data["mu0"])])
        self.sigma0 = np.array([data["sigma0"][k] for k in sorted(data["sigma0"])])
        self.obs_noise = obs_noise
        self.lever_names = sorted(data["mu0"])

    def update(self, X: np.ndarray, y: np.ndarray):
        """
        X – shape (N, J)   (features for each trade)
        y – shape (N,)      (profit contribution per trade, e.g., +1 / -1)
        Returns posterior mu, sigma2.
        """
        XtX = X.T @ X
        Xty = X.T @ y

        inv_sigma0 = np.diag(1.0 / self.sigma0)
        inv_sigma_n = inv_sigma0 + XtX / self.obs_noise
        sigma_n = np.linalg.inv(inv_sigma_n)

        mu_n = sigma_n @ (inv_sigma0 @ self.mu0 + Xty / self.obs_noise)

        return mu_n, np.diag(sigma_n)

    def save_posterior(self, mu, sigma2, out_path: Path):
        out = {
            "mu":   {k: float(v) for k, v in zip(self.lever_names, mu)},
            "sigma2":{k: float(v) for k, v in zip(self.lever_names, sigma2)},
        }
        out_path.write_text(json.dumps(out, indent=2))
