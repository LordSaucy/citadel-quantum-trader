#!/usr/bin/env python3
"""
run_opt.py

Weekly optimiser for Citadel Quantum Trader.

Runs a genetic algorithm (DEAP) or CMA-ES to tune trading parameters:
- Risk fractions per bucket
- SMC signal weights
- Entry/exit multipliers
- Regime weights

Best config is persisted to new_config.yaml, validated in sandbox,
and then promoted to production via the weekly deployment pipeline.
"""

import argparse
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from deap import algorithms, base, creator, tools
from prometheus_client import Counter, Gauge, start_http_server

from config_loader import Config
from backtest_validator import BacktestValidator
from fitness import fitness

# -----------------------------------------------------------------
# Helper: convert a flat list of numbers into the dict expected by fitness()
# -----------------------------------------------------------------
def decode_individual(ind, param_schema):
    """
    ind – list of floats produced by the optimiser
    param_schema – ordered list of (name, bounds, type) tuples
    Returns a dict compatible with fitness().
    """
    cfg = {}
    idx = 0
    for name, bounds, kind in param_schema:
        if kind == "list":                     # e.g., risk_schedule (fixed length)
            length = bounds[2]                 # third element stores length
            cfg[name] = ind[idx:idx+length]
            idx += length
        else:
            cfg[name] = ind[idx]
            idx += 1
    return cfg


# -----------------------------------------------------------------
# Define the optimisation search space
# -----------------------------------------------------------------
# (name, (lower, upper, optional_extra), kind)
#   kind = "float"  → single scalar
#   kind = "list"   → a fixed‑length list of floats
PARAM_SCHEMA = [
    ("smc_weights", (0.0, 2.0, 7), "list"),          # 7 lever weights
    ("risk_schedule", (0.0, 1.0, 7), "list"),        # 7 risk‑fractions (first 2 = 1.0)
    ("rr_target", (3.0, 7.0, None), "float"),        # 5 : 1 is typical
    ("win_rate_target", (0.95, 0.999, None), "float"),
    ("max_drawdown", (0.05, 0.20, None), "float"),
]

# Flatten the bounds for the optimiser
LOWER = []
UPPER = []
for _, (lo, hi, extra), _ in PARAM_SCHEMA:
    if extra is None:          # scalar
        LOWER.append(lo)
        UPPER.append(hi)
    else:                      # list – repeat bounds for each element
        for _ in range(extra):
            LOWER.append(lo)
            UPPER.append(hi)

LOWER = np.array(LOWER)
UPPER = np.array(UPPER)

# -----------------------------------------------------------------
# Parameter definition for DEAP
# -----------------------------------------------------------------
PARAM_KEYS = [
    "liq_imbalance_weight",
    "atr_k_stop",
    "regime_weight_trend",
    "regime_weight_range",
    "risk_fraction_1",
    "risk_fraction_2",
    "risk_fraction_3",
]

PARAM_BOUNDS = {
    "liq_imbalance_weight": (0.0, 0.30),
    "atr_k_stop": (1.0, 4.0),
    "regime_weight_trend": (0.0, 1.0),
    "regime_weight_range": (0.0, 1.0),
    "risk_fraction_1": (0.001, 0.02),
    "risk_fraction_2": (0.001, 0.02),
    "risk_fraction_3": (0.001, 0.02),
}

# -----------------------------------------------------------------
# Parse CLI args
# -----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Weekly CQT optimiser")
parser.add_argument("--method", choices=["deap", "cmaes"], default="cmaes")
parser.add_argument("--data-dir", default="../backend/data")
parser.add_argument("--generations", type=int, default=30)
parser.add_argument("--popsize", type=int, default=30)
parser.add_argument("--seed", type=int, default=int(time.time()))
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

# -----------------------------------------------------------------
# Prometheus metrics
# -----------------------------------------------------------------
optimiser_runs_total = Counter("optimiser_runs_total", "Number of optimiser executions")
optimiser_last_fitness = Gauge("optimiser_last_fitness", "Best fitness from the most recent run")
optimiser_last_duration = Gauge(
    "optimiser_last_duration_seconds",
    "Runtime of the most recent optimiser run"
)

# -----------------------------------------------------------------
# ✅ FIXED: Removed commented-out code
# (Previously had: "# take‑profit = entry + multiplier * ATR")
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Fitness evaluator
# -----------------------------------------------------------------
def evaluate(individual):
    """
    Evaluate an individual (config) on train & validation sets.
    
    Returns a tuple (fitness,) for DEAP compatibility.
    """
    # 1️⃣ Map genes → config dict
    cfg_updates = dict(zip(PARAM_KEYS, individual))

    # 2️⃣ Merge with base config (risk caps, broker creds, etc.)
    base_cfg = Config().settings.copy()
    base_cfg.update(cfg_updates)

    # 3️⃣ Fast back‑test on TRAIN slice (metrics collected but not used for fitness)
    train_df = pd.read_parquet(os.getenv("TRAIN_DATA"))
    validator = BacktestValidator(data=train_df, **base_cfg)
    # ✅ FIXED: Removed unused assignment of train_res
    validator.run()

    # 4️⃣ Full back‑test on VALIDATION slice
    val_df = pd.read_parquet(os.getenv("VAL_DATA"))
    validator_val = BacktestValidator(data=val_df, **base_cfg)
    val_res = validator_val.run()

    # 5️⃣ Primary fitness – expectancy on validation
    wr = val_res["win_rate"]
    rr = val_res["avg_rr"]
    exp = wr * rr - (1 - wr)                     # simple expectancy

    # 6️⃣ Penalties
    # a) Draw‑down > 20 % → 10 pts per % over the limit
    dd_pen = max(0, (val_res["max_dd"] - 0.20)) * 10

    # b) Risk‑per‑trade exceeds hard cap (default 1.5 %)
    risk_pen = 0
    max_risk = base_cfg.get("max_risk_per_trade", 0.015)
    if cfg_updates.get("liq_imbalance_weight", 0) > max_risk:
        risk_pen = (cfg_updates.get("liq_imbalance_weight", 0) - max_risk) * 5

    # c) Optional drift penalty (keeps parameters near defaults)
    drift_pen = 0
    for k, v in cfg_updates.items():
        default = Config().settings.get(k)
        if default is not None:
            drift_pen += abs(v - default) * 2   # weight = 2 points per unit drift

    # 7️⃣ Final fitness (higher = better)
    fitness_score = exp - dd_pen - risk_pen - drift_pen
    return (fitness_score,)


# -----------------------------------------------------------------
# DEAP boilerplate setup
# -----------------------------------------------------------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def random_individual():
    """Create a gene list respecting the bounds."""
    return creator.Individual([random.uniform(*PARAM_BOUNDS[k]) for k in PARAM_KEYS])


toolbox = base.Toolbox()
toolbox.register("individual", random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# -----------------------------------------------------------------
# Sanity check: ensure parameters are valid
# -----------------------------------------------------------------
def sanity_check(params):
    """Validate that parameters meet hard constraints."""
    # 1️⃣ Ensure risk schedule never exceeds 1.0 (100 %)
    if isinstance(params.get("risk_schedule"), list):
        if any(r > 1.0 for r in params["risk_schedule"]):
            return False, "Risk schedule > 100 %"

    # 2️⃣ Verify SMC weights are positive
    if isinstance(params.get("smc_weights"), list):
        if any(w <= 0 for w in params["smc_weights"]):
            return False, "SMC weight <= 0"

    # 3️⃣ Enforce a minimum RR (e.g., >= 3.0)
    if params.get("rr_target", 0) < 3.0:
        return False, "RR_target too low"

    # 4️⃣ Max draw‑down must be <= 0.15 (15 %)
    if params.get("max_drawdown", 0) > 0.15:
        return False, "max_drawdown > 15%"

    return True, "OK"


# -----------------------------------------------------------------
# Persist optimiser run metadata
# -----------------------------------------------------------------
def log_run(best_cfg, fitness_score, duration):
    """Write run result to SQLite database for audit trail."""
    db_path = os.path.join(os.path.dirname(__file__), "optimiser_log.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_ts TEXT,
            fitness REAL,
            config_json TEXT,
            duration_s REAL
        )
    """)
    cur.execute(
        "INSERT INTO runs (run_ts, fitness, config_json, duration_s) VALUES (?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            fitness_score,
            json.dumps(best_cfg),
            duration
        ),
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------
# Sandbox validation (before deploying to production)
# -----------------------------------------------------------------
SANDBOX_DATA = os.getenv("SANDBOX_DATA", "/data/sandbox_validation.parquet")
CURRENT_CFG_PATH = "/opt/config/current_config.yaml"


def run_sandbox(cfg_path):
    """Execute a full back‑test on held‑out sandbox data."""
    cmd = [
        "python", "backtest_runner.py",
        "--config", cfg_path,
        "--data", SANDBOX_DATA,
        "--output", "/tmp/sandbox_report.json"
    ]
    subprocess.check_call(cmd, cwd="/app")
    with open("/tmp/sandbox_report.json") as f:
        return json.load(f)


def validate_sandbox():
    """Run the new config through sandbox and compare vs current production config."""
    sandbox_res = run_sandbox("/opt/config/new_config.yaml")
    prev_res = run_sandbox(CURRENT_CFG_PATH)

    # ---- Acceptance criteria ----
    MIN_EXPECTANCY_IMPROVEMENT = 0.05   # 5 % better than current
    if sandbox_res["expectancy"] < prev_res["expectancy"] * (1 + MIN_EXPECTANCY_IMPROVEMENT):
        print("❌ Sandbox failed – not enough expectancy gain")
        return False

    if sandbox_res["max_dd"] > 0.20:
        print("❌ Sandbox failed – draw‑down > 20 %")
        return False

    print("✅ Sandbox passed – moving to paper‑trading")
    return True


# -----------------------------------------------------------------
# Main optimisation loop
# -----------------------------------------------------------------
def main():
    """Run the optimiser (DEAP or CMA-ES) and persist results."""
    start_ts = time.time()

    if args.method == "cmaes":
        # ---------
        # CMA-ES
        # ---------
        import cma

        # Initial guess = middle of the bounds
        x0 = (LOWER + UPPER) / 2.0
        sigma0 = 0.3 * (UPPER - LOWER)

        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            sigma0.tolist(),
            {
                'bounds': [LOWER.tolist(), UPPER.tolist()],
                'popsize': args.popsize,
                'seed': args.seed,
                'verb_disp': 0,
            }
        )

        best_score = -np.inf
        best_params = None

        for gen in range(args.generations):
            solutions = es.ask()
            fitness_vals = []
            for sol in solutions:
                param_dict = decode_individual(sol, PARAM_SCHEMA)
                score = fitness(param_dict, args.data_dir)
                fitness_vals.append(-score)          # CMA‑ES minimises → negate
            es.tell(solutions, fitness_vals)

            gen_best_idx = np.argmin(fitness_vals)
            gen_best_score = -fitness_vals[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_params = decode_individual(solutions[gen_best_idx], PARAM_SCHEMA)

            print(f"[Gen {gen+1:02d}] best fitness = {best_score:.4f}")

    else:
        # ---------
        # DEAP GA
        # ---------
        pop = toolbox.population(n=args.popsize)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        optimiser_runs_total.inc()

        pop, _ = algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.6, mutpb=0.3,
            ngen=args.generations,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )

        best_score = hof[0].fitness.values[0]
        best_params = dict(zip(PARAM_KEYS, hof[0]))

    # ----- Sanity check -----
    ok, msg = sanity_check(best_params)
    if not ok:
        print(f"❌ Optimiser produced invalid config: {msg}")
        return False

    # ----- Write new config -----
    out_path = Path("../config/new_config.yaml")
    final_cfg = Config().settings.copy()
    final_cfg.update(best_params)
    out_path.write_text(yaml.dump(final_cfg, default_flow_style=False))

    duration = time.time() - start_ts
    optimiser_last_fitness.set(best_score)
    optimiser_last_duration.set(duration)
    log_run(best_params, best_score, duration)

    print(f"\n✅ Optimiser finished – best fitness = {best_score:.4f}")
    print(f"📝 New config written to {out_path}")

    # ----- Sandbox validation -----
    if not validate_sandbox():
        return False

    return True


if __name__ == "__main__":
    start_http_server(8001)  # Prometheus metrics on port 8001
    success = main()
    sys.exit(0 if success else 1)
