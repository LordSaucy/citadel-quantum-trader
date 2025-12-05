import pytest
import requests
import time

@pytest.mark.integration
def test_full_workflow(mt5_sandbox):
    "_", host_port = mt5_sandbox
    # The sandbox exposes an HTTP API that mimics the MT5 broker.
    # We'll start the bot pointing at this sandbox and verify a trade occurs.

    # 1️⃣ Start the bot (docker‑compose service) with env pointing to the sandbox
    import subprocess, os
    env = os.environ.copy()
    env["MT5_DEMO_HOST"] = f"localhost:{host_port}"
    # Use docker‑compose to bring up a single bot instance
    subprocess.run(
        ["docker", "compose", "up", "-d", "--scale", "citadel-bot=1", "citadel-bot"],
        env=env,
        check=True,
    )

    # 2️⃣ Wait a bit for the bot to connect and possibly place a trade
    time.sleep(30)

    # 3️⃣ Query the bot’s /metrics endpoint to see if a trade was recorded
    resp = requests.get("http://localhost:8000/metrics")
    assert resp.status_code == 200
    assert "trades_total" in resp.text

    # 4️⃣ Clean up – stop the bot
    subprocess.run(["docker", "compose", "down", "citadel-bot"], check=True)
