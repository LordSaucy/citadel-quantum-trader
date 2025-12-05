# src/api/bot.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from datetime import datetime, timezone

from .deps import get_current_user
from ..execution_engine.execution_engine import ExecutionEngine

router = APIRouter(prefix="/bot", tags=["bot"])

# Singleton execution engine (or a façade that talks to it)
engine = ExecutionEngine()   # <-- replace with your actual init / DI

class PauseReq(BaseModel):
    reason: str | None = Field(None, description="Why you are pausing")

class KillReq(BaseModel):
    duration_seconds: int = Field(300, ge=30, description="Kill‑switch duration")
    reason: str | None = Field(None, description="Reason for kill‑switch")

class WithdrawReq(BaseModel):
    symbol: str
    percentage: float = Field(..., gt=0, le=1, description="Fraction of position to close")

@router.post("/pause", response_model=dict)
async def pause(req: PauseReq | None = None, token: str = Depends(get_current_user)):
    engine.pause(reason=req.reason if req else None)
    return {"paused": True, "since": datetime.now(timezone.utc).isoformat()+"Z", "reason": req.reason if req else None}

@router.post("/resume", response_model=dict)
async def resume(token: str = Depends(get_current_user)):
    engine.resume()
    return {"paused": False, "resumed_at": datetime.utcnow(timezone.utc).isoformat()+"Z"}

@router.post("/kill", response_model=dict)
async def kill(req: KillReq, token: str = Depends(get_current_user)):
    engine.activate_kill_switch(duration=timedelta(seconds=req.duration_seconds),
                               reason=req.reason)
    return {"kill_active": True,
            "expires_at": (datetime.now(timezone.utc)+timedelta(seconds=req.duration_seconds)).isoformat()+"Z",
            "reason": req.reason}

@router.get("/kill/status", response_model=dict)
async def kill_status(token: str = Depends(get_current_user)):
    active, expires = engine.kill_status()
    return {"kill_active": active, "expires_at": expires.isoformat()+"Z" if expires else None}

@router.post("/withdraw", response_model=dict)
async def withdraw(req: WithdrawReq, token: str = Depends(get_current_user)):
    result = engine.withdraw(symbol=req.symbol, pct=req.percentage)
    return result
