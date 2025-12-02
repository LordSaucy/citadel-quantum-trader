import sqlite3, json, time
import os, yaml, random, time, json, sqlite3
import numpy as np, pandas as pd
from datetime import datetime
from deap import base, creator, tools, algorithms
from config_loader import Config
from backtest_validator import BacktestValidator   # your existing validator
from prometheus_client import Gauge, Counter, start_http_server



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
