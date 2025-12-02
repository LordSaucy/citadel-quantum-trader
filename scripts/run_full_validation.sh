#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=./data/parquet
REPORT_DIR=./reports
CONFIG=./config/config.yaml
FEES=./config/fees.yaml

# 1️⃣ Full‑cost back‑test
python backtest.py \
  --data "${DATA_DIR}/EURUSD.parquet" \
  --config "$CONFIG" \
  --commission 2.0 \
  --spread-pips 2.0 \
  --slippage-pips 0.5 \
  --swap-file "$FEES" \
  --output "${REPORT_DIR}/backtest_EURUSD.json"

# Extract net win‑rate & net R for the next steps
WR=$(jq .net_win_rate "${REPORT_DIR}/backtest_EURUSD.json")
RR=$(jq .net_avg_R "${REPORT_DIR}/backtest_EURUSD.json")

# 2️⃣ Monte‑Carlo walk‑forward
python monte_carlo_dd.py \
  --win-rate "$WR" \
  --avg-R "$RR" \
  --trades 1000 \
  --paths 10000 \
  --output "${REPORT_DIR}/monte_carlo_EURUSD.json"

# 3️⃣ Capital‑model stress test (70 % win‑rate week)
python stress_capital.py \
  --base-wr "$WR" \
  --base-rr "$RR" \
  --drop-wr 0.70 \
  --weeks 1 \
  --output "${REPORT_DIR}/stress_drop70.json"

echo "✅ All validation steps completed. Check ${REPORT_DIR} for JSON reports."
