#!/usr/bin/env python3
"""
fetch_events.py – Pull the daily economic‑calendar from Investing.com
and upsert the rows into the CQT PostgreSQL database.

The script is intended to be run **once per day** (e.g. via systemd timer
or a cron job).  It reads the DB connection string from the environment
variable ``POSTGRES_URL`` (the same format you use for the engine).

Author: Citadel Quantum Trader team
"""

import os
import csv
import io
import sys
import logging
from datetime import datetime, timezone

import requests
import psycopg2
from psycopg2.extras import execute_batch

# -----------------------------------------------------------------
# Configuration (environment variables)
# -----------------------------------------------------------------
DB_URL = os.getenv("POSTGRES_URL")  # e.g. postgresql://cqt_user:pwd@cqt-db:5432/cqt_ledger
if not DB_URL:
    sys.stderr.write("ERROR: POSTGRES_URL environment variable not set\n")
    sys.exit(1)

INVESTING_CAL_URL = (
    "https://api.investing.com/api/financialdata/calendar?timezone=UTC"
)

# -----------------------------------------------------------------
# Logging – simple stdout logger (feel free to replace with structlog)
# -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# 1️⃣  Fetch the raw JSON payload from Investing.com
# -----------------------------------------------------------------
def fetch_events() -> list[tuple]:
    """Return a list of tuples ready for bulk insert."""
    log.info("Downloading economic‑calendar from Investing.com")
    headers = {"User-Agent": "CitadelBot/1.0"}  # required by the API
    resp = requests.get(INVESTING_CAL_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()["data"]  # list of dicts

    rows: list[tuple] = []
    for ev in payload:
        # Expected keys (may be missing → use .get()):
        #   title, date (YYYY‑MM‑DD), time (HH:MM), country, impact
        try:
            # Combine date+time and force UTC tzinfo
            ts = datetime.strptime(
                f"{ev['date']} {ev['time']}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone.utc)

            rows.append(
                (
                    ev["title"],                # event_name
                    ts,                         # event_ts_utc (datetime)
                    ev.get("country", ""),      # country (optional)
                    ev["impact"].lower(),       # impact_level (low/medium/high)
                    "investing.com",            # source
                )
            )
        except Exception as exc:
            log.warning("Skipping malformed event %s – %s", ev, exc)
            continue

    log.info("Fetched %d calendar entries", len(rows))
    return rows


# -----------------------------------------------------------------
# 2️⃣  Bulk upsert into PostgreSQL (ON CONFLICT DO NOTHING)
# -----------------------------------------------------------------
def upsert_events(rows: list[tuple]) -> None:
    """Insert rows into ``event_calendar``; ignore duplicates."""
    if not rows:
        log.info("No rows to upsert – exiting.")
        return

    log.info("Connecting to PostgreSQL")
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO event_calendar
                    (event_name, event_ts_utc, country, impact_level, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (event_name, event_ts_utc) DO NOTHING;
            """
            # `execute_batch` is faster than executemany for many rows
            execute_batch(cur, sql, rows, page_size=500)
        conn.commit()
        log.info("Successfully upserted %d rows", len(rows))
    finally:
        conn.close()


# -----------------------------------------------------------------
# 3️⃣  Main entry‑point (called by systemd / cron)
# -----------------------------------------------------------------
if __name__ == "__main__":
    try:
        events = fetch_events()
        upsert_events(events)
        log.info("Economic‑calendar refresh completed")
        sys.exit(0)
    except Exception as exc:
        log.exception("Fatal error while refreshing calendar: %s", exc)
        sys.exit(1)
