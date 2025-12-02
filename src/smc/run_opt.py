import sqlite3, datetime, json

# --------------------------------------------------------------
# 1️⃣ Parameter keys (add SMC knobs)
# --------------------------------------------------------------
PARAM_KEYS = [
    "risk_pct", "sl_multiplier", "tp_multiplier", "smc_weight",
    # NEW SMC-specific knobs
    "order_block_weight",
    "liquidity_sweep_weight",
    "fair_value_gap_weight",
    "break_of_structure_weight",
    "premium_discount_thresh",
    "max_confluence_score",
]

# --------------------------------------------------------------
# 2️⃣ Bounds for each knob (DEAP expects a tuple (low, high))
# --------------------------------------------------------------
PARAM_BOUNDS = {
    "risk_pct": (0.3, 2.0),                     # % of aggressive pool
    "sl_multiplier": (0.5, 2.0),                # ATR‑scaled stop‑loss factor
    "tp_multiplier": (1.0, 4.0),                # ATR‑scaled TP factor
    "smc_weight": (0.5, 0.9),                   # overall SMC contribution
    # NEW SMC knobs
    "order_block_weight": (0.0, 1.0),
    "liquidity_sweep_weight": (0.0, 1.0),
    "fair_value_gap_weight": (0.0, 1.0),
    "break_of_structure_weight": (0.0, 1.0),
    "premium_discount_thresh": (0.0, 0.05),
    "max_confluence_score": (0.8, 1.0),
}

def log_run(best_vals, fitness):
    db_path = "/opt/optimizer/optimiser_history.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            ts TEXT PRIMARY KEY,
            fitness REAL,
            params TEXT
        )
    """)
    cur.execute(
        "INSERT OR REPLACE INTO runs (ts, fitness, params) VALUES (?,?,?)",
        (datetime.datetime.utcnow().isoformat(),
         fitness,
         json.dumps(best_vals))
    )
    conn.commit()
    conn.close()


def write_new_config(best_vals: dict):
    # Load the base config (the one you already use)
    with open("/opt/optimizer/base_config.yaml") as f:
        base_cfg = yaml.safe_load(f)

    # Overwrite the tuned sections
    base_cfg["risk_fraction"] = best_vals["risk_pct"] / 100.0   # example conversion
    base_cfg["sl_multiplier"] = best_vals["sl_multiplier"]
    base_cfg["tp_multiplier"] = best_vals["tp_multiplier"]
    base_cfg["smc_weight"] = best_vals["smc_weight"]

    # NEW – SMC specific knobs
    base_cfg.setdefault("smc_parameters", {})
    for key in [
        "order_block_weight",
        "liquidity_sweep_weight",
        "fair_value_gap_weight",
        "break_of_structure_weight",
        "premium_discount_thresh",
        "max_confluence_score",
    ]:
        base_cfg["smc_parameters"][key] = best_vals[key]

    # Persist
    with open("/opt/optimizer/new_config.yaml", "w") as out:
        yaml.safe_dump(base_cfg, out, sort_keys=False)
WINDOW_DAYS = int(os.getenv("OPT_WINDOW_DAYS", "30"))

def load_recent_data():
    # Assuming you store raw tick/minute data in /data/
    # Use pandas to slice the last WINDOW_DAYS
    import pandas as pd
    df = pd.read_parquet("/data/full_history.parquet")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=WINDOW_DAYS)
    return df[df["timestamp"] >= cutoff]

# In the evaluation function:
def evaluate(individual):
    cfg_updates = dict(zip(PARAM_KEYS, individual))
    # Load a *fresh* window each evaluation (cheap because data is cached in RAM)
    recent_df = load_recent_data()
    # Pass recent_df to BacktestValidator instead of the full dataset
    validator = BacktestValidator(data=recent_df, **base_cfg)
    # … rest unchanged …


 
