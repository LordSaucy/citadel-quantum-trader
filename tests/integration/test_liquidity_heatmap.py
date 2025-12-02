import os
import tempfile
import pytest
from unittest.mock import MagicMock
from src.position_stacking_manager import PositionStackingManager
from src.utils.heatmap import plot_depth_heatmap

@pytest.fixture
def fake_broker():
    broker = MagicMock()
    # Simulate a shallow order book (tiny bid volume)
    broker.get_market_depth.return_value = [
        {"price": 1.1000, "bid_vol": 10, "ask_vol": 5000},
        {"price": 1.1005, "bid_vol": 15, "ask_vol": 4800},
        {"price": 1.1010, "bid_vol": 12, "ask_vol": 4700},
    ]
    return broker

def test_check_liquidity_triggers_heatmap(fake_broker, tmp_path, monkeypatch):
    # Force the heat‑map output directory to a temporary location
    monkeypatch.setattr("src.utils.heatmap.os.makedirs", lambda *_: None)
    monkeypatch.setattr("src.utils.heatmap.plt.savefig",
                        lambda path, **_: open(path, "wb").write(b"pngdata"))

    psm = PositionStackingManager(broker=fake_broker)
    # Request a volume larger than the total bid side (10+15+12 = 37)
    result = psm._check_liquidity(symbol="EURUSD", required_volume=100, side="buy")
    assert result is False

    # Verify that the broker was asked for depth
    fake_broker.get_market_depth.assert_called_once_with("EURUSD", depth=30)

    # Verify that a PNG file was created (the mocked save writes dummy data)
    heatmap_files = list(tmp_path.parent.glob("heatmaps/*.png"))
    assert heatmap_files, "Heat‑map PNG was not generated"
