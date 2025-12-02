# src/risk_bandit.py
import numpy as np
import math
import json
from pathlib import Path

class RiskBandit:
    """
    Thompson‑sampling bandit that learns a multiplicative aggressiveness factor α.
    Stores the posterior of log(α) as (mu, sigma2).
    """
    def __init__(self, prior_path: Path, sigma_obs: float = 0.2):
        """
        prior_path – JSON with {"mu": 0.0, "sigma2": 0.5}
        sigma_obs – assumed observation noise on log‑reward.
        """
        if prior_path.exists():
            data = json.loads(prior_path.read_text())
            self.mu    = data["mu"]
            self.sigma2 = data["sigma2"]
        else:
            self.mu    = 0.0   # log(1.0) => α = 1.0
            self.sigma2 = 1.0

        self.sigma_obs2 = sigma_obs ** 2
        self.prior_path = prior_path

    def sample_alpha(self) -> float:
        """Draw a sample from the current posterior and exponentiate."""
        log_alpha = np.random.normal(self.mu, math.sqrt(self.sigma2))
        return math.exp(log_alpha)

    def update(self, reward: float):
        """
        `reward` – realised profit divided by the *scheduled* risk amount.
        Positive reward > 1 means the trade outperformed the schedule,
        reward < 1 means it under‑performed.
        We work in log‑space:
            log_reward = log(reward)
        """
        if reward <= 0:
            # Defensive: avoid log(0) – treat as a strong negative signal
            log_r = -5.0
        else:
            log_r = math.log(reward)

        # Conjugate Gaussian update for log(α)
        precision_prior = 1.0 / self.sigma2
        precision_obs   = 1.0 / self.sigma_obs2

        self.sigma2 = 1.0 / (precision_prior + precision_obs)
        self.mu    = self.sigma2 * (precision_prior * self.mu + precision_obs * log_r)

    def persist(self):
        data = {"mu": self.mu, "sigma2": self.sigma2}
        self.prior_path.write_text(json.dumps(data, indent=2))
