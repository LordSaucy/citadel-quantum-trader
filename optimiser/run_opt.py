import sqlite3, json, time

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
