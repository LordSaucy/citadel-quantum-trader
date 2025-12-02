import aiohttp
import os

BASE_URL = os.getenv("BOT_CONTROL_URL", "http://bot-control:8000")
HEADERS = {"Content-Type": "application/json"}

async def _post(endpoint: str, payload: dict = None):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}{endpoint}",
            json=payload or {},
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"{endpoint} failed: {resp.status} {txt}")
            return await resp.json()


async def pause_all():
    return await _post("/api/v1/pause")


async def resume_all():
    return await _post("/api/v1/resume")


async def kill_switch():
    return await _post("/api/v1/kill-switch")
