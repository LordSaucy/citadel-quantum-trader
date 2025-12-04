# scripts/evaluate_paper.py
import json, sys
with open("paper_report.json") as f:
    r = json.load(f)

# Example thresholds
if r["order_reject_rate"] > 0.01:
    sys.exit("❌ Too many order rejections")
if r["avg_slippage"] > 0.5:   # pips
    sys.exit("❌ Slippage exceeds limit")
if r["expectancy"] < 0.0:
    sys.exit("❌ Expectancy negative")
print("✅ Paper‑trading passed")
