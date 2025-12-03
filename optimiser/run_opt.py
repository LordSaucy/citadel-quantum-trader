import sqlite3, json, time
import os, yaml, random, time, json, sqlite3
import numpy as np, pandas as pd
from datetime import datetime
from deap import base, creator, tools, algorithms
from config_loader import Config
from backtest_validator import BacktestValidator   # your existing validator
from prometheus_client import Gauge, Counter, start_http_server
import argparse, json, os, sys, time
from pathlib import Path
from fitness import fitness
import numpy as np
from datetime import datetime
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


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
    ("rr_target", (3.0, 7.0, None), "float"),       # 5 : 1 is typical
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
# Parse CLI args
# -----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Weekly CQT optimiser")
parser.add_argument("--method", choices=["deap","cmaes"], default="cmaes")
parser.add_argument("--data-dir", default="../backend/data")
parser.add_argument("--generations", type=int, default=30)
parser.add_argument("--popsize", type=int, default=30)
parser.add_argument("--seed", type=int, default=int(time.time()))
args = parser.parse_args()

np.random.seed(args.seed)

# -----------------------------------------------------------------
# 1️⃣  CMA‑ES implementation (fast for continuous spaces)
# -----------------------------------------------------------------
if args.method == "cmaes":
    import cma

    # Initial guess = middle of the bounds
    x0 = (LOWER + UPPER) / 2.0
    sigma0 = 0.3 * (UPPER - LOWER)   # initial step size

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0.tolist(),
                                  {'bounds': [LOWER.tolist(), UPPER.tolist()],
                                   'popsize': args.popsize,
                                   'seed': args.seed,
                                   'verb_disp': 0})   # silence verbose output

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

        # Keep track of the best (largest) score we have seen
        gen_best_idx = np.argmin(fitness_vals)   # because we negated
        gen_best_score = -fitness_vals[gen_best_idx]
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_params = decode_individual(solutions[gen_best_idx], PARAM_SCHEMA)

        print(f"[Gen {gen+1:02d}] best fitness = {best_score:.4f}")

    # -----------------------------------------------------------------
    # 2️⃣  Write the winning config to new_config.yaml
    # -----------------------------------------------------------------
    out_path = Path("../config/new_config.yaml")
    out_cfg = {
        "smc_weights": best_params["smc_weights"],
        "risk_schedule": {
            1: best_params["risk_schedule"][0],
            2: best_params["risk_schedule"][1],
            3: best_params["risk_schedule"][2],
            4: best_params["risk_schedule"][3],
            5: best_params["risk_schedule"][4],
            6: best_params["risk_schedule"][5],
            7: best_params["risk_schedule"][6],
        },
        "RR_target": best_params["rr_target"],
        "win_rate_target": best_params["win_rate_target"],
        "max_drawdown": best_params["max_drawdown"],
        # Preserve everything else from the current config:
        # (you can merge with the existing file if you like)
    }

    out_path.write_text(json.dumps(out_cfg, indent=2))
    print(f"\n✅ Optimiser finished – best score {best_score:.4f}")
    print(f"📝 New config written to {out_path}")
elif args.method == "deap":
    # -------------------------------------------------
    # DEAP GA setup (already partially defined above)
    # -------------------------------------------------
    from deap import algorithms

    # Statistics collector – useful for debugging / logging
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg",   np.mean)
    stats.register("std",   np.std)
    stats.register("min",   np.min)
    stats.register("max",   np.max)

    # Run the evolutionary loop
    pop, logbook = algorithms.eaSimple(population=pop,
                                        toolbox=toolbox,
                                        cxpb=0.7,          # crossover probability
                                        mutpb=0.2,         # mutation probability
                                        ngen=args.generations,
                                        stats=stats,
                                        halloffame=hof,
                                        verbose=False)

    # -------------------------------------------------
    # Extract the best individual from the Hall‑of‑Fame
    # -------------------------------------------------
    best_ind = hof[0]
    best_score = best_ind.fitness.values[0]
    best_params = decode_individual(best_ind, PARAM_SCHEMA)

    print("\n=== DEAP RESULT ===")
    print(f"Best fitness: {best_score:.4f}")
    print(f"Best individual: {best_params}")

    # -------------------------------------------------
    # Write the winning configuration to new_config.yaml
    # -------------------------------------------------
    out_path = Path("../config/new_config.yaml")
    out_cfg = {
        "smc_weights": best_params["smc_weights"],
        "risk_schedule": {
            1: best_params["risk_schedule"][0],
            2: best_params["risk_schedule"][1],
            3: best_params["risk_schedule"][2],
            4: best_params["risk_schedule"][3],
            5: best_params["risk_schedule"][4],
            6: best_params["risk_schedule"][5],
            7: best_params["risk_schedule"][6],
        },
        "RR_target": best_params["rr_target"],
        "win_rate_target": best_params["win_rate_target"],
        "max_drawdown": best_params["max_drawdown"],
    }

    out_path.write_text(json.dumps(out_cfg, indent=2))
    print(f"\n✅ Optimiser finished – best score {best_score:.4f}")
    print(f"📝 New config written to {out_path}")




def log_run(best_cfg, fitness, duration):
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
        (datetime.utcnow().isoformat(), fitness, json.dumps(best_cfg), duration),
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    start = time.time()
    best_cfg, best_fit = run_optimiser()
    duration = time.time() - start
    log_run(best_cfg, best_fit, duration)
    # write new_config.yaml as before …
# ---------- 1️⃣ Parameter definition ----------
PARAM_KEYS = [
    "risk_pct",          # % of equity to risk per trade (0‑1)
    "sl_multiplier",    # stop‑loss = entry – multiplier * ATR
    "tp_multiplier",    # take‑profit = entry + multiplier * ATR
    "smc_weight",       # weight of the SMC lever (0‑1)
    # add any other levers you expose in config.yaml
]

PARAM_BOUNDS = {
    "risk_pct":       (0.003, 0.020),   # 0.3 % – 2 %
    "sl_multiplier": (0.5, 3.0),
    "tp_multiplier": (1.0, 5.0),
    "smc_weight":    (0.5, 0.9),
    # extend as needed
}

PARAM_KEYS = [
    "liq_imbalance_weight",
    "atr_k_stop",
    "regime_weight_trend",
    "regime_weight_range",
    # existing keys …
    "risk_fraction_1",
    "risk_fraction_2",
    "risk_fraction_3",
    # …
]

PARAM_BOUNDS = {
    # New knobs -------------------------------------------------
    "liq_imbalance_weight": (0.0, 0.30),   # 0 %‑30 % of the SMC score
    "atr_k_stop": (1.0, 4.0),            # 1‑4 × ATR stop‑loss
    "regime_weight_trend": (0.0, 1.0),    # 0‑100 % emphasis on trend regime
    "regime_weight_range": (0.0, 1.0),    # 0‑100 % emphasis on range regime
    # Existing knobs (example) -------------------------------
    "risk_fraction_1": (0.001, 0.02),
    "risk_fraction_2": (0.001, 0.02),
    # …
}


# ---------- 2️⃣ DEAP boilerplate ----------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def random_individual():
    """Create a gene list respecting the bounds."""
    return creator.Individual([random.uniform(*PARAM_BOUNDS[k]) for k in PARAM_KEYS])

toolbox = base.Toolbox()
toolbox.register("individual", random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ---------- 3️⃣ Prometheus metrics ----------
optimiser_runs_total   = Counter("optimiser_runs_total", "Number of optimiser executions")
optimiser_last_fitness = Gauge("optimiser_last_fitness", "Best fitness from the most recent run")
optimiser_last_duration = Gauge("optimiser_last_duration_seconds",
                               "Runtime of the most recent optimiser run")

# ---------- 4️⃣ Fitness evaluator ----------
def evaluate(individual):
    # 1️⃣ Map genes → config dict
    cfg_updates = dict(zip(PARAM_KEYS, individual))

    # 2️⃣ Merge with base config (risk caps, broker creds, etc.)
    base_cfg = Config().settings.copy()
    base_cfg.update(cfg_updates)

    # 3️⃣ Fast back‑test on TRAIN slice
    train_df = pd.read_parquet(os.getenv("TRAIN_DATA"))
    validator = BacktestValidator(data=train_df, **base_cfg)
    train_res = validator.run()

    # 4️⃣ Full back‑test on VALIDATION slice
    val_df = pd.read_parquet(os.getenv("VAL_DATA"))
    validator_val = BacktestValidator(data=val_df, **base_cfg)
    val_res = validator_val.run()

    # 5️⃣ Primary fitness – expectancy on validation
    wr = val_res["win_rate"]
    rr = val_res["avg_rr"]
    exp = wr * rr - (1 - wr)                     # simple expectancy

    # 6️⃣ Penalties
    # a) Draw‑down > 20 % → 10 pts per % over the limit
    dd_pen = max(0, (val_res["max_dd"] - 0.20)) * 10

    # b) Risk‑per‑trade exceeds hard cap (default 1.5 %)
    risk_pen = 0
    max_risk = base_cfg.get("max_risk_per_trade", 0.015)
    if cfg_updates["risk_pct"] > max_risk:
        risk_pen = (cfg_updates["risk_pct"] - max_risk) * 5

    # c) Optional drift penalty (keeps parameters near defaults)
    drift_pen = 0
    for k, v in cfg_updates.items():
        default = Config().settings.get(k)
        if default is not None:
            drift_pen += abs(v - default) * 2   # weight = 2 points per unit drift

    # 7️⃣ Final fitness (higher = better)
    fitness = exp - dd_pen - risk_pen - drift_pen
    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------- 8️⃣ Persist optimiser run metadata ----------
def log_run(best_cfg, fitness, duration):
    db_path = os.path.join(os.path.dirname(__file__), "optimiser_log.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_ts TEXT,
        fitness REAL,
        config_json TEXT,
        duration_s REAL
    )""")
    cur.execute(
        "INSERT INTO runs (run_ts, fitness, config_json, duration_s) VALUES (?,?,?,?)",
        (datetime.utcnow().isoformat(), fitness, json.dumps(best_cfg), duration),
    )
    conn.commit()
    conn.close()

# ---------- 9️⃣ Main optimisation loop ----------
def main(pop_size=30, ngen=15, seed=42):
    random.seed(seed)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)          # best individual ever seen

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    start = time.time()
    optimiser_runs_total.inc()

    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.6, mutpb=0.3,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    duration = time.time() - start
    best_fitness = hof[0].fitness.values[0]

    # ---- Record metrics & DB log ----
    optimiser_last_fitness.set(best_fitness)
    optimiser_last_duration.set(duration)
    log_run(dict(zip(PARAM_KEYS, hof[0])), best_fitness, duration)

    # ---- Write the best config for the bot ----
    best_cfg = dict(zip(PARAM_KEYS, hof[0]))
    final_cfg = Config().settings.copy()
    final_cfg.update(best_cfg)

    out_path = os.getenv("OUT_CONFIG", "/opt/config/new_config.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(final_cfg, f)

    print(f"\n✅ Optimiser finished – best fitness = {best_fitness:.4f}")
    print(f"📝 New config written to {out_path}")

if __name__ == "__main__":
    # Expose Prometheus metrics on a side‑port (8001)
    start_http_server(8001)
    main()
import sqlite3, uuid, json, datetime

def persist_best(run_id, cfg, fitness, metrics, notes=""):
    conn = sqlite3.connect("optimiser_history.db")
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS history (
        run_id TEXT PRIMARY KEY,
        timestamp TEXT,
        config_json TEXT,
        fitness REAL,
        metrics TEXT,
        notes TEXT
    )""")
    cur.execute(
        "INSERT INTO history (run_id, timestamp, config_json, fitness, metrics, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_id,
            datetime.datetime.utcnow().isoformat(),
            json.dumps(cfg),
            fitness,
            json.dumps(metrics),
            notes,
        ),
    )
    conn.commit()
    conn.close()

registry = CollectorRegistry()
g_fitness = Gauge("cqt_opt_best_fitness", "Best fitness per generation", registry=registry)
g_runtime = Gauge("cqt_opt_runtime_seconds", "Total optimiser runtime", registry=registry)

# Inside the generation loop:
g_fitness.set(best_score)

# After the loop:
g_runtime.set(time.time() - start_ts)
push_to_gateway("pushgateway.mycompany.internal:9091", job="cqt_optimiser", registry=registry)

def sanity_check(params):
    # 1️⃣ Ensure risk schedule never exceeds 1.0 (100 %)
    if any(r > 1.0 for r in params["risk_schedule"]):
        return False, "Risk schedule > 100 %"

    # 2️⃣ Verify SMC weights are positive (negative weights would invert signals)
    if any(w <= 0 for w in params["smc_weights"]):
        return False, "SMC weight <= 0"

    # 3️⃣ Enforce a minimum RR (e.g., >= 3.0)
    if params["rr_target"] < 3.0:
        return False, "RR_target too low"

    # 4️⃣ Max draw‑down must be <= 0.15 (15 %) – we never want a config that allows >15 %
    if params["max_drawdown"] > 0.15:
        return False, "max_drawdown > 15%"

    return True, "OK"

ok, msg = sanity_check(best_params)
if not ok:
    raise RuntimeError(f"Optimiser produced invalid config: {msg}")
