# src/api/deps.py
from fastapi import Depends, HTTPException, Header, status
from typing import Annotated

# In production you would store the token in a secret manager.
# For simplicity we read it from an environment variable.
import os
CQT_API_TOKEN = os.getenv("CQT_API_TOKEN")
if not CQT_API_TOKEN:
    raise RuntimeError("CQT_API_TOKEN env var not set!")

def get_current_user(authorization: Annotated[str | None, Header(alias="Authorization")] = None):
    """
    Simple bearer‑token auth used by every internal endpoint.
    Expected header:  Authorization: Bearer <token>
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != CQT_API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid token")
    return token   # you could return a user object here if you wish
