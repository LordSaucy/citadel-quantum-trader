@app.post("/api/v1/kill_switch")
def kill_switch(payload: dict):
    # Verify the request came from Alertmanager (shared secret or IP whitelist)
    # Set a flag in Redis / DB that the main loop checks each iteration.
    redis.set("kill_switch_active", "1")
    logger.warning("Monte‑Carlo DD monitor triggered kill‑switch")
    return {"status": "killed"}
