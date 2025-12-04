# src/api/config.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, Dict

from .deps import get_current_user

router = APIRouter(prefix="/config", tags=["confluence"])

# In‑memory store – replace with Redis / DB in production
_TUNABLES: Dict[str, Any] = {
    "weight_mtf_structure": 0.20,
    "weight_aoi": 0.15,
    "weight_candlestick": 0.10,
    "weight_smc": 0.12,
    "weight_head_shoulders": 0.08,
    # … add the rest of your tunables …
}

class ConfigUpdate(BaseModel):
    value: Any = Field(..., description="New value for the tunable")

@router.get("/", response_model=dict)
async def dump_all(token: str = Depends(get_current_user)):
    """Return the full dictionary of tunables."""
    return {"config": _TUNABLES}

@router.get("/{key}", response_model=dict)
async def get_key(key: str, token: str = Depends(get_current_user)):
    if key not in _TUNABLES:
        raise HTTPException(status_code=404, detail=f"Tunable {key} not found")
    return {"key": key, "value": _TUNABLES[key]}

@router.post("/{key}", response_model=dict)
async def set_key(key: str, payload: ConfigUpdate, token: str = Depends(get_current_user)):
    old = _TUNABLES.get(key)
    _TUNABLES[key] = payload.value
    # TODO: persist to Redis / DB and broadcast change via Pub/Sub if needed
    return {"key": key, "old": old, "new": payload.value}
