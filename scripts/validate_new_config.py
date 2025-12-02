#!/usr/bin/env python3
import yaml, os, sys, pandas as pd
from backtest_validator import BacktestValidator
from config_loader import Config

# -----------------------------------------------------------------
# Paths – adjust if you mount the data elsewhere
# -----------------------------------------------------------------
CONFIG_PATH = "/opt/config/new_config.yaml"
VALIDATION_DATA = "/data/validation.parquet"   # a recent 3‑month slice

# -----------------------------------------------------------------
# Load the candidate config
# -----------------------------------------------------------------
if not os.path.isfile(CONFIG_PATH):
    sys.exit("[ERROR] new_config.yaml not found")

with open(CONFIG_PATH) as f:
    candidate_cfg = yaml.safe_load(f)

# Merge with the rest of the base config (broker creds, DB, etc.)
base_cfg = Config().settings.copy()
base_cfg.update(candidate_cfg)

# -----------------------------------------------------------------
# Run a *full* back‑test (no shortcuts)
# -----------------------------------------------------------------
df = pd.read_parquet(VALIDATION_DATA)
validator = BacktestValidator(data=df, **base_cfg)
res = validator.run()

# -----------------------------------------------------------------
# Acceptance criteria (tweak to your risk appetite)
# -----------------------------------------------------------------
MIN_EXPECTANCY = 0.02          # 2 % expectancy per trade (example)
MAX_DRAWDOWN   = 0.18          # 18 % max draw‑down on the validation slice
MAX_SLIPPAGE   = 0.3           # 0.3 pips average slippage allowed

issues = []

exp = res["win_rate"] * res["avg_rr"] - (1 - res["win_rate"])
if exp < MIN_EXPECTANCY:
    issues.append(f"❌ Expectancy {exp:.3f} < {MIN_EXPECTANCY}")

if res["max_dd"] > MAX_DRAWDOWN:
    issues.append(f"❌ Draw‑down {res['max_dd']:.2%} > {MAX_DRAWDOWN:.2%}")

if res.get("avg_slippage", 0) > MAX_SLIPPAGE:
    issues.append(f"❌ Avg slippage {res['avg_slippage']:.2f} pips > {MAX_SLIPPAGE}")

if issues:
    print("\n".join(issues))
    sys.exit(1)

print("[OK] Validation passed – config is safe to roll out.")
print(f"  Expectancy: {exp:.3f}")
print(f"  Max DD:    {res['max_dd']:.2%}")
print(f"  Avg slip:  {res.get('avg_slippage',0):.2f} pips")
