### 8. News‑Sentiment & External‑Shock Guard

1. Verify the sentiment ingest container is healthy:
   - `docker logs citadel-sentiment-ingestor` → should show “Fetched 0 items” every minute.
   - Prometheus metric `sentiment_freshness_seconds` should be < 30 s.

2. Verify the calendar lock‑out:
   - `docker exec citadel-bot python -c "from src.guards.calendar_lockout import CalendarLockout; print(CalendarLockout(...).is_locked())"`
   - In Grafana, watch the “Calendar Lock‑out (active?)” stat panel.

3. If any guard fires unexpectedly:
   - Check the corresponding Prometheus counter (e.g., `sentiment_guard_hits_total`).
   - Review the bot logs for the guard‑specific INFO/WARN messages.
   - If the guard is mis‑configured, edit `config.yaml` and trigger a hot‑reload (`docker exec citadel-bot kill -HUP 1` or use the Admin UI Config Editor).

4. Emergency bypass:
   - Set `guards.sentinel.guard_enabled = false` in `config.yaml` (or use the Admin UI toggle) and reload.
   - **Only** do this if you are certain the guard is malfunctioning; otherwise you expose the bot to uncontrolled news risk.
