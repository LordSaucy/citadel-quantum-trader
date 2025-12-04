import pandas as pd
df = pd.read_csv('trades_2024_12_01.csv')
# assume columns: trade_id, symbol, entry_price, exit_price, profit_loss, lir, blocked (bool)
passed = df[~df['blocked']]
blocked = df[df['blocked']]
for grp, name in [(passed, 'PASSED'), (blocked, 'BLOCKED'), (df, 'ALL')]:
    win_rate = (grp['profit_loss'] > 0).mean()
    sharpe   = grp['profit_loss'].mean() / grp['profit_loss'].std()
    print(f"{name}: trades={len(grp)} win%={win_rate:.2%} sharpe={sharpe:.2f}")
Interpret the results
