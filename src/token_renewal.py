# -------------------------------------------------
# token_renewal.py – runs in a daemon thread
# -------------------------------------------------
import os
import threading
import time
import hvac
import logging
import sys

log = logging.getLogger(__name__)

def _renew_token_periodically(vault_addr: str, token: str, interval_secs: int = 12 * 3600):
    """
    Calls `vault token renew -increment=24h` every `interval_secs`.
    If renewal fails three times in a row the process exits (watchdog will restart it).
    """
    client = hvac.Client(url=vault_addr, token=token)
    failures = 0
    while True:
        try:
            client.auth.token.renew_self(increment="24h")
            log.info("✅ Vault token successfully renewed (24 h increment)")
            failures = 0
        except Exception as exc:   # pragma: no cover – only hits on real failure
            failures += 1
            log.error(f"❌ Vault token renewal failed ({failures}/3): {exc}")
            if failures >= 3:
                log.critical("💥 Vault token could not be renewed – exiting so watchdog restarts")
                sys.exit(1)
        time.sleep(interval_secs)


def start_renewal_thread():
    """Call once after the ConfigLoader has obtained VAULT_ADDR & VAULT_TOKEN."""
    vault_addr = os.getenv("VAULT_ADDR")
    vault_token = os.getenv("VAULT_TOKEN")
    if not vault_addr or not vault_token:
        log.warning("Vault address or token not set – token renewal thread not started")
        return

    t = threading.Thread(
        target=_renew_token_periodically,
        args=(vault_addr, vault_token),
        daemon=True,
        name="vault-token-renewal",
    )
    t.start()
# -----------------------------------------------------------------
# Initialise the singleton and start the renewal thread
# -----------------------------------------------------------------
_cfg = Config()
# Kick‑off the background renewal (runs forever)
from .token_renewal import start_renewal_thread
start_renewal_thread()
