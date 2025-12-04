#!/usr/bin/env python3
"""
Creates a configurable stream of fake orders against the bot’s public API.
Useful for driving the custom‑metric HPA (orders_per_sec) or the CPU HPA.
"""

import argparse, asyncio, aiohttp, random, time

API_URL = "https://admin.citadel.local/api/v1/order"   # goes through Nginx → FastAPI → bot
TOKEN = ""   # set via env var or CLI arg

async def send_one(session):
    payload = {
        "symbol": random.choice(["EURUSD", "GBPUSD", "XAUUSD"]),
        "volume": 0.01,
        "side": random.choice(["buy", "sell"]),
        "price": None,   # let the bot pick the market price
    }
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    async with session.post(API_URL, json=payload, headers=headers) as resp:
        await resp.text()   # ignore response; just fire‑and‑forget

async def main(rate, duration):
    async with aiohttp.ClientSession() as session:
        start = time.time()
        while time.time() - start < duration:
            tasks = [send_one(session) for _ in range(rate)]
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)   # one‑second bucket

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--rate", type=int, default=20,
                   help="Orders per second to generate")
    p.add_argument("-d", "--duration", type=int, default=180,
                   help="How many seconds to run")
    p.add_argument("-t", "--token", default="", help="JWT/Okta token")
    args = p.parse_args()
    TOKEN = args.token
    asyncio.run(main(args.rate, args.duration))
