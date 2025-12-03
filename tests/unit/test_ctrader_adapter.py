import pytest
from src.ctrader_adapter import CTraderAdapter

@pytest.fixture
def cfg():
    return {
        "ctrader": {
            "api_url": "https://sandbox.api.ctrader.com/v2",
            "api_key": "dummy-key"
        }
    }

def test_connect_success(monkeypatch, cfg):
    # monkeypatch the requests Session to return a successful ping
    class DummyResp:
        ok = True
        def raise_for_status(self): pass
    monkeypatch.setattr("requests.sessions.Session.get", lambda *a, **kw: DummyResp())
    adapter = CTraderAdapter(cfg)
    assert adapter.connect() is True
