# admin_ui/backend/app/main.py
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import os

# -----------------------------------------------------------------
# Import internal helpers (keep the imports tidy)
# -----------------------------------------------------------------
from .auth import get_current_user
from .bot_control import pause_all, resume_all, kill_switch
from .config import load_config, save_config
from .logs import stream_logs
from .prometheus import query_drawdown
from .triangular_arb_executor import execute_triangular_arb, ArbExecutionError
from src.bot_control import set_bucket_flag
from datetime import datetime, timezone

app = FastAPI(title="Citadel Admin API", version="0.1.0")

# -----------------------------------------------------------------
# CORS – only allow the UI origin (served from the same host)
# -----------------------------------------------------------------
origins = [f"https://{os.getenv('ADMIN_UI_HOST', 'admin.citadel.local')}"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# Health endpoint (used by watchdog)
# -----------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# -----------------------------------------------------------------
# Bot control
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# Config handling
# -----------------------------------------------------------------
@app.get("/config")
async def get_cfg(user=Depends(get_current_user)):
    cfg = await load_config()
    return cfg

@app.put("/config")
async def update_cfg(
    new_cfg: dict,
    bg: BackgroundTasks,
    user=Depends(get_current_user),
):
    if "risk_schedule" not in new_cfg:
        raise HTTPException(status_code=400, detail="Missing required key: risk_schedule")
    await save_config(new_cfg)
    # Example: pause → reload → resume (you can replace with a real reload call)
    bg.add_task(pause_all)
    return {"msg": "Config saved and will be re‑loaded"}

# -----------------------------------------------------------------
# Log streaming (plain HTTP)
# -----------------------------------------------------------------
@app.get("/logs/{container_name}")
async def logs_endpoint(container_name: str, user=Depends(get_current_user)):
    return stream_logs(container_name)

# -----------------------------------------------------------------
# Server‑Sent Events (SSE) log tail – **moved out of drawdown**
# -----------------------------------------------------------------
log_router = APIRouter()


@log_router.get("/api/logs/{container_name}")
async def sse_logs(container_name: str, user=Depends(get_current_user)):
    async def event_generator():
        async for line in log_generator(container_name):
            yield f"data: {line}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


app.include_router(log_router)

# -----------------------------------------------------------------
# Draw‑down chart data (simple JSON response)
# -----------------------------------------------------------------
@app.get("/drawdown")
async def drawdown(
    start: str | None = None,
    end: str | None = None,
    user=Depends(get_current_user),
):
    # `query_drawdown` already returns a list of points
    data = query_drawdown(start, end)
    return {"points": data}

# -----------------------------------------------------------------
# Manual triangular arbitrage execution
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
        return {"status": "success", "msg": "Arb executed successfully"}
    except ArbExecutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

# -----------------------------------------------------------------
# Bucket schedule toggle (admin endpoint)
# -----------------------------------------------------------------
admin_router = APIRouter(prefix="/admin")


@admin_router.post("/bucket/{bucket_id}/schedule")
async def toggle_schedule(
    bucket_id: int,
    enabled: bool,
    user=Depends(get_current_user),
):
    """Enable or disable the risk‑schedule for a specific bucket."""
    set_bucket_flag(bucket_id, enabled)  # writes to Redis or a DB table
    return {"bucket_id": bucket_id, "schedule_enabled": enabled}


app.include_router(admin_router)

# -----------------------------------------------------------------
# Drift‑monitor trigger (background task)
# -----------------------------------------------------------------
from fastapi import BackgroundTasks
from src.drift_monitor import run_drift_check
import pandas as pd


@app.post("/drift/run")
async def trigger_drift(background: BackgroundTasks, user=Depends(get_current_user)):
    """Pull the latest feature dataframe and run the drift calculation in the background."""
    async def _task():
        # TODO: replace with a real DB query that returns a DataFrame
        df = pd.DataFrame()  # placeholder
        if not df.empty:
            run_drift_check(df)

    background.add_task(_task)
    return {"msg": "Drift check scheduled"}

# -----------------------------------------------------------------
# Config reload endpoint (debug helper)
# -----------------------------------------------------------------
@app.post("/config/reload")
async def reload_config(user=Depends(get_current_user)):
    from config_loader import Config
    Config()._load()
    return {"msg": "Configuration reloaded from config.yaml"}
