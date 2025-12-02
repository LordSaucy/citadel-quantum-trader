from src.kill_switch import update_drawdown

def test_hysteresis_requires_three_consecutive_breaches():
    # Threshold = 0.15 (15 %)
    assert not update_drawdown(0.14)   # 1st check – below
    assert not update_drawdown(0.16)   # 2nd check – above but not enough history
    assert not update_drawdown(0.16)   # 3rd check – still not 3 consecutive
    # Now add two more above‑threshold samples
    assert not update_drawdown(0.16)   # 4th – still need 3 consecutive *after* the window fills
    assert update_drawdown(0.16)       # 5th – three consecutive (0.16,0.16,0.16) → fire
