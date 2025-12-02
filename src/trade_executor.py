# Inside src/trade_executor.py (pseudo‑code)
from .risk_bandit import RiskBandit
bandit = RiskBandit(Path('state/risk_bandit.json'))

def on_trade_settled(trade):
    # `trade` contains: scheduled_risk (dollar), realised_pnl (dollar)
    reward = (trade.realised_pnl + trade.scheduled_risk) / trade.scheduled_risk
    # Example: win = +5R → reward = (5*R + 1*R) / 1*R = 6.0
    bandit.update(reward)
    bandit.persist()

def compute_effective_risk(bucket_id, equity):
    # 1️⃣ Get the schedule fraction (e.g., 0.4 for trade #5)
    schedule_f = schedule_lookup(bucket_id)   # existing function
    # 2️⃣ Sample α from the bandit
    alpha = bandit.sample_alpha()
    # 3️⃣ Apply a safety clamp (never go below 0.5× schedule, never above 2×)
    alpha = max(0.5, min(alpha, 2.0))
    return equity * schedule_f * alpha
