# src/market_feed.py
import asyncio
from src.telemetry import trace

async def market_feed():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("market_feed"):
        # Existing market‑feed logic (e.g., websocket, REST poll)
        await asyncio.sleep(0)   # placeholder for real work
        # If you call external HTTP APIs, instrument them with
        # opentelemetry‑instrumentation‑requests (installed via opentelemetry‑instrumentation)
