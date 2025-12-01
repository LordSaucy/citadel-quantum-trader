import os
import logging
from typing import Any

# ----------------------------------------------------------------------
# Logging – one global logger used by every module
# ----------------------------------------------------------------------
LOG_LEVEL = os.getenv("CQT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cqt")

# ----------------------------------------------------------------------
# Helper to read env vars with type‑casting & fallback
# ----------------------------------------------------------------------
def env(name: str, default: Any = None, cast: type = str) -> Any:
    """Read an environment variable, cast it, and return a default on missing."""
    raw = os.getenv(name, default)
    try:
        return cast(raw) if raw is not None else default
    except Exception as exc:  # pragma: no cover
        logger.error("Invalid env %s=%s (expected %s): %s", name, raw, cast, exc)
        raise
