import logging
import datetime
from prometheus_client import Counter, Gauge
import pytz

log = logging.getLogger("citadel.calendar_lockout")

calendar_lockout_active = Gauge(
    "calendar_lockout_active",
    "1 when a high‑impact calendar window is open, else 0"
)

lockout_hits = Counter(
    "calendar_lockout_hits_total",
    "Number of times a trade was blocked by the calendar lock‑out"
)

class CalendarLockout:
    """
    Uses the `economic_calendar.py` module (already present in the repo) to
    determine whether the current UTC time falls inside a “high‑impact” window.
    The calendar module should expose a function:
        get_upcoming_events() -> List[dict] with keys:
            start_utc, end_utc, importance (e.g. "high")
    """

    def __init__(self, cfg, calendar_module):
        """
        cfg – expects:
            calendar:
                lockout_enabled: true
                lockout_margin_minutes: 30   # minutes before/after the event
        calendar_module – the imported `economic_calendar` python module.
        """
        self.enabled = cfg.get("calendar", {}).get("lockout_enabled", True)
        self.margin = datetime.timedelta(
            minutes=cfg.get("calendar", {}).get("lockout_margin_minutes", 30)
        )
        self.cal = calendar_module

    def _now_utc(self):
        return datetime.datetime.now().replace(tzinfo=pytz.UTC)

    def is_locked(self) -> bool:
        if not self.enabled:
            calendar_lockout_active.set(0)
            return False

        now = self._now_utc()
        for ev in self.cal.get_upcoming_events():
            # Expect `ev["start_utc"]` and `ev["end_utc"]` as aware datetime objects
            start = ev["start_utc"] - self.margin
            end = ev["end_utc"] + self.margin
            if start <= now <= end and ev.get("importance") == "high":
                calendar_lockout_active.set(1)
                lockout_hits.inc()
                log.info("[CalendarLockout] Blocking trade – %s", ev["title"])
                return True

        calendar_lockout_active.set(0)
        return False
