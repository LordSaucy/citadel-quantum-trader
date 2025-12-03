import psycopg2, os
from datetime import datetime, timedelta, timezone

DB_URL = os.getenv('POSTGRES_URL')
WINDOW_PRE  = int(os.getenv('EVENT_PRE_MINUTES', 5))
WINDOW_POST = int(os.getenv('EVENT_POST_MINUTES', 15))

def get_active_event(now: datetime):
    """
    Returns the highest‑impact event that overlaps the current time,
    or None if we are outside any window.
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT event_name, impact_level, event_ts_utc
        FROM event_calendar
        WHERE event_ts_utc BETWEEN %s AND %s
        ORDER BY CASE impact_level
                 WHEN 'high'   THEN 3
                 WHEN 'medium' THEN 2
                 WHEN 'low'    THEN 1
                 END DESC
        LIMIT 1;
    """, (now - timedelta(minutes=WINDOW_PRE),
          now + timedelta(minutes=WINDOW_POST)))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row   # (name, impact, ts) or None
