# Business Continuity Plan – Citadel Quantum Trader
*Version 1.2 – {{DATE}}*

## 1. Critical Components & RTO/RPO
| Component | RTO | RPO | Primary host |
|-----------|-----|-----|--------------|
| Vault (secrets) | 2 min | 0 min (replicated) | `vps-primary` |
| PostgreSQL (ledger) | 5 min | 1 h (hourly snapshot) | `vps-primary` |
| Bot containers (2 buckets) | 1 min | 0 min (stateless) | `vps-primary` |
| Watchdog (fail‑over) | 30 s | – | `vps-primary` |
| Grafana / Prometheus | 2 min | 5 min (metrics retention) | `vps-primary` |

## 2. Immediate Fail‑over Procedure
1. **Detect failure** – Watchdog logs “primary unreachable” → Slack alert.  
2. **Move floating IP** to standby VPS:  
   ```bash
   doctl compute floating-ip-action assign <FLOATING_IP> --droplet <STANDBY_DROPLET_ID>
