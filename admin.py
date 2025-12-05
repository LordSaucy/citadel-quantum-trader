# src/api/admin.py
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
import subprocess
import os
from datetime import datetime, timezone

from .deps import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])

class InfoResponse(BaseModel):
    version: str
    uptime: str
    git_sha: Optional[str] = None
    python_version: str

@router.get("/info", response_model=InfoResponse)
async def info(token: str = Depends(get_current_user)):
    # Uptime – read from the container’s start time (approx.)
    start_ts = datetime.datetime.fromtimestamp(
        os.stat("/proc/1").st_ctime
    )
    uptime = datetime.datetime.utcnow(datetime.timezone.utc) - start_ts
    # Git SHA – you can embed it at build time via an ARG in Dockerfile
    git_sha = os.getenv("GIT_SHA", "unknown")
    return InfoResponse(
        version=os.getenv("CQT_VERSION", "dev"),
        uptime=str(uptime).split(".")[0],
        git_sha=git_sha,
        python_version=os.getenv("PYTHON_VERSION", "3.12")
    )

class ReloadResponse(BaseModel):
    reloaded: bool
    message: str

@router.post("/reload", response_model=ReloadResponse)
async def reload_config(background_tasks: BackgroundTasks,
                       token: str = Depends(get_current_user)):
    """
    Trigger a graceful restart of the engine container.
    The actual restart is performed in the background so the HTTP
    request returns immediately.
    """
    def _restart():
        subprocess.run(
            ["docker", "compose", "restart", "engine"],
            check=False,
        )
    background_tasks.add_task(_restart)
    return ReloadResponse(reloaded=True,
                          message="Engine restart scheduled")
