#!/usr/bin/env bash
# -------------------------------------------------
# weekly_optimiser_runner.sh
# Runs the DEAP/CMA‑ES optimiser every Sunday at 02:00 UTC.
# -------------------------------------------------

# Absolute path to the optimiser directory (must match your volume mount)
OPT_DIR="/opt/citadel/optimizer"

cd "$OPT_DIR" || {
    echo "Cannot cd to $OPT_DIR – abort"
    exit 1
}

# Run the optimiser (the script must be present and executable)
python3 run_opt.py

# If a new config was produced, log it with a timestamp
if [[ -f new_config.yaml ]]; then
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$ts – Optimiser produced new_config.yaml (fitness: $(grep -i fitness new_config.yaml))" \
        >> /var/log/citadel/optimiser.log
fi
