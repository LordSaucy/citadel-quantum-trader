#!/usr/bin/env python3
"""
Rotate MT5 and PostgreSQL credentials in Vault.
Intended to be run by a CI job (weekly) or manually.
"""

import os
import hvac
import secrets
import string
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VAULT_ADDR = os.getenv("VAULT_ADDR")
VAULT_TOKEN = os.getenv("VAULT_TOKEN")
if not VAULT_ADDR or not VAULT_TOKEN:
    log.error("VAULT_ADDR and VAULT_TOKEN must be exported")
    sys.exit(1)

client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)

def _gen_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def rotate_mt5():
    """Generate a fresh MT5 user/password pair and store in KV."""
    new_user = f"user_{secrets.token_hex(4)}"
    new_pass = _gen_password()
    secret_path = "secret/data/citadel/mt5"
    client.secrets.kv.v2.create_or_update_secret(
        path=secret_path,
        secret={"mt5_user": new_user, "mt5_pass": new_pass},
    )
    log.info(f"✅ Rotated MT5 credentials → {secret_path}")
    return new_user, new_pass


def rotate_postgres():
    """Generate a fresh PostgreSQL password."""
    new_pass = _gen_password()
    secret_path = "secret/data/citadel/postgres"
    client.secrets.kv.v2.create_or_update_secret(
        path=secret_path,
        secret={"postgres_user": "citadel", "postgres_pass": new_pass},
    )
    log.info(f"✅ Rotated PostgreSQL password → {secret_path}")
    return new_pass


def main():
    rotate_mt5()
    rotate_postgres()
    # Optional: force a token renewal so the new secrets are picked up ASAP
    client.auth.token.renew_self(increment="24h")
    log.info("🔄 Token renewed after rotation")


if __name__ == "__main__":
    main()
