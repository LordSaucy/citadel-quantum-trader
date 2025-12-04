import json, re, sys
shadow_file = "shadow.log"
metrics = {"orders":0, "wins":0, "pnl":0.0}
with open(shadow_file) as f:
    for line in f:
        if "[SHADOW]" not in line: continue
        # extract P&L from the log line (you must format it in the engine)
        m = re.search(r"P&L=([-+]?\d+\.\d+)", line)
        if m:
            pnl = float(m.group(1))
            metrics["pnl"] += pnl
            metrics["orders"] += 1
            if pnl > 0: metrics["wins"] += 1

if metrics["orders"] == 0:
    sys.exit("❌ No shadow orders recorded")
win_rate = metrics["wins"]/metrics["orders"]
expectancy = metrics["pnl"]/metrics["orders"]
print(f"Shadow win‑rate: {win_rate:.2%}, expectancy: {expectancy:.4f}")

# Apply the same acceptance thresholds you used for paper‑trading
if win_rate < 0.55:
    sys.exit("❌ Shadow win‑rate too low")
if expectancy < 0.0:
    sys.exit("❌ Shadow expectancy negative")
print("✅ Shadow passed")
