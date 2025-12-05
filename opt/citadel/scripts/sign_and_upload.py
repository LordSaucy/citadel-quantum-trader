#!/usr/bin/env python3
"""
Nightly job – sign the latest ledger snapshot with an HSM‑backed KMS key
and upload both the snapshot and a JSON manifest containing:
    * S3 object key of the snapshot
    * SHA‑256 Merkle root (hex)
    * Base64‑encoded KMS signature
    * UTC timestamp of signing
"""

import os
import sys
import json
import base64
import datetime
import logging
from pathlib import Path
from utils import sha256_hex, kms_sign, s3_put

# ------------------------------------------------------------------
# Logging configuration (sent to stdout → captured by Docker → Loki)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('cqt.signing')

# ------------------------------------------------------------------
# Environment variables (injected by Docker / ECS task)
# ------------------------------------------------------------------
LEDGER_DIR      = Path(os.getenv('LEDGER_DIR', '/opt/citadel/ledger'))
SNAPSHOT_PREFIX = os.getenv('SNAPSHOT_PREFIX', 'ledger_snapshot_')
S3_BUCKET       = os.getenv('AUDIT_S3_BUCKET', 'citadel-audit')
KMS_SIGN_KEY    = os.getenv('KMS_SIGNING_KEY')          # e.g. arn:aws:kms:…
# Optional: separate KMS key for S3 server‑side encryption
# (set AWS_KMS_S3_KEY env var if you want a distinct key)

# ------------------------------------------------------------------


def find_latest_snapshot() -> Path:
    """Return the newest snapshot file (lexicographically highest)."""
    candidates = sorted(LEDGER_DIR.glob(f'{SNAPSHOT_PREFIX}*.json'))
    if not candidates:
        raise FileNotFoundError(f'No snapshots found in {LEDGER_DIR}')
    latest = candidates[-1]
    log.info("Latest snapshot: %s", latest.name)
    return latest


def build_manifest(snapshot_key: str, merkle_root: str, signature_b64: str) -> bytes:
    """Create the JSON manifest that will be uploaded alongside the snapshot."""
    manifest = {
        "snapshot_key": snapshot_key,
        "merkle_root": merkle_root,
        "signature_b64": signature_b64,
        "signed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    return json.dumps(manifest, separators=(',', ':')).encode('utf-8')


def main():
    """
    Main workflow:
    1. Find latest snapshot
    2. Compute SHA-256 hash (Merkle root)
    3. Sign with KMS
    4. Upload snapshot to S3
    5. Upload manifest to S3
    """
    try:
        # 1️⃣ Locate the most recent hourly snapshot
        snap_path = find_latest_snapshot()

        # 2️⃣ Compute its SHA‑256 hash (this is the Merkle root)
        merkle_root = sha256_hex(str(snap_path))
        log.info("Computed Merkle root: %s", merkle_root)

        # 3️⃣ Ask KMS (HSM‑backed) to sign the hash
        signature_raw = kms_sign(KMS_SIGN_KEY, merkle_root)
        signature_b64 = base64.b64encode(signature_raw).decode('ascii')
        log.info("Obtained KMS signature (%d bytes)", len(signature_raw))

        # 4️⃣ Upload the snapshot itself (object key = filename)
        snapshot_key = f'snapshots/{snap_path.name}'
        with open(snap_path, 'rb') as f:
            # ✅ FIXED: Single, clean call with no duplicate parameters
            s3_put(
                bucket=S3_BUCKET,
                key=snapshot_key,
                body=f.read(),
                content_type='application/json'
            )

        # 5️⃣ Build and upload the manifest (signature file)
        manifest_body = build_manifest(snapshot_key, merkle_root, signature_b64)
        manifest_key = f'signatures/{snap_path.stem}_manifest.json'
        # ✅ FIXED: Single, clean call with no duplicate parameters
        s3_put(
            bucket=S3_BUCKET,
            key=manifest_key,
            body=manifest_body,
            content_type='application/json'
        )

        log.info("Nightly signing job completed successfully.")
        return 0

    except Exception as exc:
        # Critical failure – push a Slack alert (or PagerDuty) and exit non‑zero
        log.exception("Signing job FAILED")
        # Simple webhook example (replace with your real alert endpoint)
        webhook = os.getenv('ALERT_WEBHOOK_URL')
        if webhook:
            try:
                import requests
                payload = {
                    "text": f":rotating_light: *CQT signing job failed* – {exc}",
                    "attachments": [{"color": "danger"}]
                }
                requests.post(webhook, json=payload, timeout=5)
            except Exception:
                pass
        sys.exit(1)


if __name__ == '__main__':
    main()
