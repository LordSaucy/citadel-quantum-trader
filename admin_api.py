from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

app = FastAPI(
    title="Citadel Quantum Trader – Admin API",
    version="1.0.0",
    description="Administrative endpoints for on‑call engineers. All calls require a TOTP‑protected bearer token."
)

security = HTTPBearer()

def verify_token(creds: HTTPAuthorizationCredentials = Security(security)):
    # Example: token is a JWT signed with the Vault signing key
    # In production you would validate the signature, expiry, and TOTP claim.
    if not creds.credentials.startswith("cqt-"):
        raise HTTPException(status_code=403, detail="Invalid token")
    return creds.credentials

@app.post("/api/v1/pause_all", tags=["Control"])
def pause_all(token: str = Depends(verify_token)):
    """Pause every bucket (sets risk_fraction to 0)."""
    # call internal service that flips a Redis flag
    return {"status": "paused"}

@app.post("/api/v1/resume_all", tags=["Control"])
def resume_all(token: str = Depends(verify_token)):
    """Resume trading for all buckets."""
    return {"status": "running"}

@app.post("/api/v1/kill_switch/reset", tags=["Safety"])
def reset_kill_switch(token: str = Depends(verify_token)):
    """Clear the kill‑switch flag after a draw‑down event."""
    return {"status": "kill‑switch cleared"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
