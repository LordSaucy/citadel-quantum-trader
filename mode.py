# src/api/mode.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from enum import Enum

from .deps import get_current_user

router = APIRouter(prefix="/mode", tags=["mode"])

class EngineMode(str, Enum):
    live   = "live"
    paper  = "paper"
    shadow = "shadow"

# In a real deployment you would store the mode in a shared config store
# (Redis, Consul, etc.).  For the demo we keep it in a module‑level var.
CURRENT_MODE: EngineMode = EngineMode.paper

class ModeResponse(BaseModel):
    mode: EngineMode

@router.get("/", response_model=ModeResponse)
async def get_mode(token: str = Depends(get_current_user)):
    return ModeResponse(mode=CURRENT_MODE)

@router.post("/", response_model=ModeResponse)
async def set_mode(new_mode: EngineMode, token: str = Depends(get_current_user)):
    global CURRENT_MODE
    if new_mode == CURRENT_MODE:
        return ModeResponse(mode=CURRENT_MODE)

    # Changing mode usually requires a container restart.
    # Here we just flip the variable; the Docker‑Compose file reads it
    # from an env‑file or Docker secret at container start.
    CURRENT_MODE = new_mode
    # Optionally trigger a restart via Docker‑Compose:
    # subprocess.run(["docker", "compose", "restart", "engine"])
    return ModeResponse(mode=CURRENT_MODE)
