# bot_control/main.py  (new file or extend existing)
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import redis   # or any KV store you already use for flags
import logging

app = FastAPI(title="Citadel Bot Control", version="0.1.0")
log = logging.getLogger("bot_control")

# -------------------------------------------------
# Simple bearer‑token auth (reuse the same secret as admin‑backend)
# -------------------------------------------------
TOKEN = os.getenv("CONTROL_API_KEY", "CHANGE_ME")
bearer_scheme = HTTPBearer(auto_error=True)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid control token")
    return True

# -------------------------------------------------
# Shared flag store – you can use Redis, a file, or the DB.
# Here we use a tiny Redis key called "kill_switch".
# -------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
r = redis.from_url(REDIS_URL)

KILL_SWITCH_KEY = "kill_switch_active"

def set_kill_switch(active: bool):
    r.set(KILL_SWITCH_KEY, int(active))

def is_kill_switch_active() -> bool:
    val = r.get(KILL_SWITCH_KEY)
    return bool(int(val or 0))

# -------------------------------------------------
# Public endpoint (protected by the bearer token)
# -------------------------------------------------
@app.post("/api/v1/kill-switch", dependencies=[Depends(verify_token)])
async def kill_switch():
    """Activate the global kill‑switch – all bots will pause immediately."""
    set_kill_switch(True)
    log.warning("🔴 Kill‑switch ACTIVATED via API")
    return {"status": "kill-switch activated"}

# -------------------------------------------------
# Helper endpoint – useful for health checks / UI status
# -------------------------------------------------
@app.get("/api/v1/kill-switch/status", dependencies=[Depends(verify_token)])
async def kill_switch_status():
    return {"active": is_kill_switch_active()}
