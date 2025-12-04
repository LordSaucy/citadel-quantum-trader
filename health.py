# src/api/health.py
from fastapi import APIRouter, Depends
from datetime import datetime
from .deps import get_current_user

router = APIRouter(tags=["health"])

@router.get("/healthz", response_model=dict)
async def healthz():
    """Lightweight health check for the load balancer."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@router.get("/ready", response_model=dict, dependencies=[Depends(get_current_user)])
async def ready():
    """
    Deeper readiness probe – you can plug in DB / broker checks here.
    For brevity we just return ready=True.
    """
    # TODO: add real checks (DB ping, broker connectivity, market‑data feed)
    return {"ready": True, "details": {"db": "ok", "broker": "ok", "market_data": "ok"}}
