# Citadel Quantum Trader – Configuration Reference

All runtime parameters are read from `config/config.yaml`.  
The file is loaded at startup and re‑loaded automatically whenever it changes.

| Section | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| `risk_schedule` | `1`, `2`, `3`, … `default` | float (0‑1) | 1.0, 1.0, 0.6, 0.5, 0.4 | Fraction of equity risked per trade (index = trade number). |
| `max_drawdown_pct` | – | float | 0.15 | Kill‑switch triggers when draw‑down ≥ 15 %. |
| `reserve_pool_pct` | – | float | 0.20 | Portion of equity kept untouched for emergencies. |
| `slippage_pips` | – | float | 0.5 | Maximum tolerated slippage before a trade is rejected. |
| `win_rate_target` | – | float | 0.997 | Target win‑rate used by the optimiser. |
| `RR_target` | – | float | 5.0 | Desired reward‑to‑risk ratio (used by optimiser). |
| `brokers` | list of objects | – | – | Each entry contains `name` and `vault_path` – credentials are pulled from Vault at runtime. |
| `alerts` | `drawdown_pct`, `latency_ms`, `winrate_floor` | – | – | Thresholds for Grafana/Alertmanager alerts. |
| `secret_backend` | `vault` / `aws` / `none` | string | `vault` | Where secrets are fetched from. |
| … (add any other keys you introduced) | | | | |

**How to edit**  
*Via UI*: Open **Config → Edit** in the Admin Console, modify the YAML, click **Save & Reload**.  
*Via CLI*: `./scripts/reload_config.py` after you manually edit the file on disk.  

Any change is applied **instantly** – the bot’s watcher picks up the new values and the next trade will use the updated parameters.
