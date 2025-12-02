import random, time
from locust import HttpUser, task, between

class BotUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def normal_order(self):
        # a realistic‑looking order payload
        payload = {
            "symbol": "EURUSD",
            "volume": random.choice([0.01, 0.02, 0.05]),
            "price": round(random.uniform(1.0800, 1.0900), 5),
            "side": random.choice(["buy", "sell"]),
        }
        self.client.post("/api/v1/test-order", json=payload)

    @task(1)
    def latency_spike(self):
        # simulate a network stall of 200‑1000 ms
        time.sleep(random.uniform(0.2, 1.0))
        self.normal_order()

    @task(2)
    def reject_burst(self):
        # send a batch of malformed orders that the bot will reject
        for _ in range(20):
            payload = {"symbol": "INVALID", "volume": 0}
            self.client.post("/api/v1/test-order", json=payload)
