#!/usr/bin/env python3
import subprocess, time, os, pathlib, pandas as pd
from datetime import datetime

DOCKER_COMPOSE = 'docker compose'   # or 'docker-compose' on older versions
RESULTS_CSV = pathlib.Path('experiments/results.csv')
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

def start_stack():
    subprocess.run(f"{DOCKER_COMPOSE} up -d", shell=True, check=True)

def stop_stack():
    subprocess.run(f"{DOCKER_COMPOSE} down -v", shell=True, check=True)

def wait_for_trades(target_trades=5000, poll_interval=5):
    """Poll the two schemas until each has at least target_trades rows."""
    import psycopg2
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='citadel',
        user='citadel',
        password=os.getenv('POSTGRES_PASSWORD')
    )
    cur = conn.cursor()
    while True:
        cur.execute("SELECT COUNT(*) FROM public.trades")
        ctrl = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM exp.trades")
        exp = cur.fetchone()[0]
        print(f"[{datetime.now()}] control={ctrl}, variant={exp}")
        if ctrl >= target_trades and exp >= target_trades:
            break
        time.sleep(poll_interval)
    cur.close()
    conn.close()

def analyse_and_append():
    import psycopg2
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='citadel',
        user='citadel',
        password=os.getenv('POSTGRES_PASSWORD')
    )
    # Pull raw data for each schema
    df_ctrl = pd.read_sql("SELECT * FROM public.trades", conn)
    df_exp  = pd.read_sql("SELECT * FROM exp.trades", conn)

    # Helper to compute the metrics you care about
    def metrics(df):
        pnl = df['pnl'].sum()
        equity = df['equity_before'].iloc[-1] + pnl
        win_rate = (df['pnl'] > 0).mean()
        sharpe = (df['pnl'].mean() / df['pnl'].std()) * (252**0.5)   # daily → annualized
        max_dd = (df['equity_before'] - df['equity_before'].cummax()).min()
        expectancy = pnl / len(df)   # avg $ per trade
        return dict(expectancy=expectancy,
                    win_rate=win_rate,
                    sharpe=sharpe,
                    max_dd=max_dd)

    m_ctrl = metrics(df_ctrl)
    m_exp  = metrics(df_exp)

    # Append a row to the CSV
    row = {
        'timestamp': datetime.now().isoformat(),
        'experiment': os.getenv('EXPERIMENT'),   # e.g. WITH_LIR
        'control_expectancy': m_ctrl['expectancy'],
        'variant_expectancy': m_exp['expectancy'],
        'delta_expectancy_%': (m_exp['expectancy'] - m_ctrl['expectancy']) / abs(m_ctrl['expectancy']) * 100,
        'control_sharpe': m_ctrl['sharpe'],
        'variant_sharpe': m_exp['sharpe'],
        'delta_sharpe': m_exp['sharpe'] - m_ctrl['sharpe'],
        'control_max_dd': m_ctrl['max_dd'],
        'variant_max_dd': m_exp['max_dd'],
        'delta_max_dd_%': (m_exp['max_dd'] - m_ctrl['max_dd']) / abs(m_ctrl['max_dd']) * 100,
        'control_win_rate': m_ctrl['win_rate'],
        'variant_win_rate': m_exp['win_rate'],
        'delta_win_rate_%': (m_exp['win_rate'] - m_ctrl['win_rate']) * 100,
    }
    df_row = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df_row.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        df_row.to_csv(RESULTS_CSV, mode='w', header=True, index=False)

    conn.close()

def run_one_experiment():
    try:
        start_stack()
        # Give the feed a moment to warm‑up
        time.sleep(5)
        wait_for_trades(target_trades=5000)
        analyse_and_append()
    finally:
        stop_stack()

if __name__ == '__main__':
    run_one_experiment()

import matplotlib.pyplot as plt
import seaborn as sns

def make_plot(row: dict):
    """Create a side‑by‑side bar chart for the most important KPIs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics = ['expectancy', 'sharpe', 'max_dd', 'win_rate']
    ctrl_vals = [row[f'control_{m}'] for m in metrics]
    exp_vals  = [row[f'variant_{m}']  for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, ctrl_vals, width, label='Control')
    ax.bar(x + width/2, exp_vals,  width, label='Variant')
    ax.set_xticks(x)
    ax.set_xticklabels(['Exp.', 'Sharpe', 'Max‑DD', 'Win‑Rate'])
    ax.set_ylabel('Metric value')
    ax.set_title(f"{row['experiment']} – {row['timestamp'][:10]}")
    ax.legend()
    plt.tight_layout()

    out_dir = pathlib.Path('experiment/results/plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{row['timestamp'][:10]}_{row['experiment']}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Plot saved to {fname}")

