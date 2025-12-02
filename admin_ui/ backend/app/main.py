from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .auth import get_current_user
from .bot_control import pause_all, resume_all, kill_switch
from .config import load_config, save_config
from .logs import stream_logs
from .prometheus import query_drawdown
import os
from fastapi import Body
from .triangular_arb_executor import execute_triangular_arb, ArbExecutionError
from fastapi import APIRouter, Depends, HTTPException
from src.bot_control import set_bucket_flag   # you’ll implement this helper


app = FastAPI(title="Citadel Admin API", version="0.1.0")

# -------------------------------------------------
# CORS – only allow the UI origin (served from the same host)
# -------------------------------------------------
origins = [
    f"https://{os.getenv('ADMIN_UI_HOST', 'admin.citadel.local')}"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Health endpoint (used by watchdog)
# -------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------------------------------
# Bot control -------------------------------------------------
@app.post("/bot/pause")
async def api_pause(user=Depends(get_current_user)):
    await pause_all()
    return {"msg": "All buckets paused"}

@app.post("/bot/resume")
async def api_resume(user=Depends(get_current_user)):
    await resume_all()
    return {"msg": "All buckets resumed"}

@app.post("/bot/kill")
async def api_kill(user=Depends(get_current_user)):
    await kill_switch()
    return {"msg": "Kill‑switch activated"}

# -------------------------------------------------
# Config handling -------------------------------------------------
@app.get("/config")
async def get_cfg(user=Depends(get_current_user)):
    cfg = await load_config()
    return cfg

@app.put("/config")
async def update_cfg(new_cfg: dict, bg: BackgroundTasks,
                    user=Depends(get_current_user)):
    # Basic validation – you can add a JSON‑Schema validator here
    if "risk_schedule" not in new_cfg:
        raise HTTPException(status_code=400,
                            detail="Missing required key: risk_schedule")
    await save_config(new_cfg)
    # Fire‑and‑forget a reload request to the bot (optional)
    bg.add_task(pause_all)   # example: pause, reload, resume
    return {"msg": "Config saved and will be re‑loaded"}

# -------------------------------------------------
# Log streaming -------------------------------------------------
@app.get("/logs/{container_name}")
async def logs_endpoint(container_name: str,
                        user=Depends(get_current_user)):
    return stream_logs(container_name)

# -------------------------------------------------
# Draw‑down chart data -------------------------------------------------
@app.get("/drawdown")
async def drawdown(start: str = None, end: str = None,
                  user=Depends(get_current_user)):

                      from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

@app.get("/api/logs/{container_name}")
async def sse_logs(container_name: str, user=Depends(get_current_user)):
    async def event_generator():
        async for line in _log_generator(container_name):
            yield f"data: {line}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

    data = query_drawdown(start, end)
    return {"points": data}

@app.post("/config/reload")
async def reload_config(user=Depends(get_current_user)):
    from config_loader import Config
    Config()._load()
    return {"msg": "Configuration reloaded from config.yaml"}

# -----------------------------------------------------------------
@app.post("/arb/run")
async def run_manual_arb(
    payload: dict = Body(...),  # expects {"legs": [...], "gross_profit_pips": float}
    user=Depends(get_current_user),
):
    """
    Payload example:
    {
        "legs": [
            {"symbol": "EURUSD", "side": "buy",  "volume": 0.01},
            {"symbol": "USDJPY", "side": "sell", "volume": 0.01},
            {"symbol": "EURJPY", "side": "sell", "volume": 0.01}
        ],
        "gross_profit_pips": 1.2
    }
    """
    try:
        await execute_triangular_arb(
            broker=bot_control,  # reuse the same broker instance you use for live trading
            legs=payload["legs"],
            gross_profit_pips=payload["gross_profit_pips"],
        )
        # Increment success metric (already done inside executor)
        return {"status": "success", "msg": "Arb executed successfully"}
    except ArbExecutionError as exc:
        # Increment guard‑hit metric (already done inside executor)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

router = APIRouter(prefix="/admin")

@router.post("/bucket/{bucket_id}/schedule")
async def toggle_schedule(bucket_id: int, enabled: bool, user=Depends(get_current_user)):
    """
    Enable or disable the risk‑schedule for a specific bucket.
    """
    set_bucket_flag(bucket_id, enabled)   # writes to Redis or a DB table
    return {"bucket_id": bucket_id, "schedule_enabled": enabled}


