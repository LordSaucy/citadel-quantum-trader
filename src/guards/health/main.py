@app.get("/health")
async def health():
    # Sentiment freshness
    raw = redis_client.get("sentiment:latest_ts")   # store timestamp when ingesting
    now = time.time()
    fresh = raw and (now - float(raw)) < 30
    return {
        "status": "ok",
        "sentiment_fresh": fresh,
        "calendar_ok": calendar_lockout.enabled,   # simple flag
    }
