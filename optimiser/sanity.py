#!/usr/bin/env python3
"""
sanity.py – post‑optimisation sanity checks for CQT parameter sets.
"""

from __future__ import annotations
from typing import Tuple, Mapping, Sequence

def sanity_check(params: Mapping[str, object]) -> Tuple[bool, str]:
    """
    Validate a dictionary of optimisation results.

    Returns
    -------
    (bool, str)
        *True*  – the parameter set is acceptable.
        *False* – the set is rejected; the second element contains a human‑readable reason.
    """
    # -------------------------------------------------------------
    # 1️⃣  Risk schedule must never exceed 100 % (1.0)
    # -------------------------------------------------------------
    risk_sched: Sequence[float] = params.get("risk_schedule", [])
    if any(r > 1.0 for r in risk_sched):
        return False, "Risk schedule > 100 %"

    # -------------------------------------------------------------
    # 2️⃣  SMC (Smart Money Concepts) weights must be strictly positive
    # -------------------------------------------------------------
    smc_weights: Sequence[float] = params.get("smc_weights", [])
    if any(w <= 0 for w in smc_weights):
        return False, "SMC weight <= 0"

    # -------------------------------------------------------------
    # 3️⃣  Minimum Reward‑to‑Risk (RR) target – we never accept < 3.0
    # -------------------------------------------------------------
    rr_target: float = float(params.get("rr_target", 0.0))
    if rr_target < 3.0:
        return False, f"RR_target too low ({rr_target:.2f} < 3.0)"

    # -------------------------------------------------------------
    # 4️⃣  Maximum draw‑down must be ≤ 15 % (0.15)
    # -------------------------------------------------------------
    max_dd: float = float(params.get("max_drawdown", 0.0))
    if max_dd > 0.15:
        return False, f"max_drawdown too high ({max_dd:.2%} > 15 %)"

    # -------------------------------------------------------------
    # 5️⃣  (Optional) Add any extra domain‑specific guards here
    # -------------------------------------------------------------
    # Example: ensure the sum of all bucket risk fractions = 1.0
    # bucket_fracs = params.get("bucket_fractions", [])
    # if abs(sum(bucket_fracs) - 1.0) > 0.001:
    #     return False, "Bucket fractions do not sum to 1.0"

    return True, "OK"
