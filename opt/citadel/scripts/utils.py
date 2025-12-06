# =============================================================================
# opt/citadel/scripts/utils.py
# =============================================================================
"""
Utility helpers used by the signing / upload workflow.

*   SHA‑256 hashing of a file
*   RSA‑PKCS#1 v1.5 signing via AWS KMS
*   Secure S3 upload (with ExpectedBucketOwner enforced)
*   End‑to‑end “hash → sign → upload” orchestration
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

# -------------------------------------------------------------------------
# Local imports – the S3 helper enforces ExpectedBucketOwner (security hotspot)
# -------------------------------------------------------------------------
from src.aws.s3_client import get_s3_client

# -------------------------------------------------------------------------
# Module‑level logger
# -------------------------------------------------------------------------
log = logging.getLogger(__name__)

# =============================================================================
# 1️⃣  SHA‑256 helper
# =============================================================================
def sha256_hex(file_path: str) -> str:
    """
    Return the SHA‑256 hex digest of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal string of the SHA‑256 digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# 2️⃣  KMS signing helper
# =============================================================================
def kms_sign(key_id: str, digest_hex: str) -> bytes:
    """
    Ask AWS KMS (or CloudHSM via KMS) to sign a SHA‑256 digest.

    Args:
        key_id: AWS KMS key ID or ARN.
        digest_hex: SHA‑256 digest as a hex string.

    Returns:
        Raw signature bytes.

    Raises:
        ClientError: If KMS signing fails.
    """
    client = boto3.client("kms")
    try:
        resp = client.sign(
            KeyId=key_id,
            Message=bytes.fromhex(digest_hex),
            MessageType="DIGEST",
            SigningAlgorithm="RSASSA_PKCS1_V1_5_SHA_256",
        )
        return resp["Signature"]
    except ClientError as exc:
        log.error("KMS sign failed: %s", exc)
        raise


# =============================================================================
# 3️⃣  Secure S3 upload helper
# =============================================================================
def s3_put(
    bucket: str,
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    """
    Upload an object to an S3 bucket using the hardened ``get_s3_client``.

    Args:
        bucket: S3 bucket name.
        key: Object key / path inside the bucket.
        data: Raw bytes to upload.
        content_type: MIME type (defaults to ``application/octet-stream``).

    Raises:
        Exception: Propagates any error raised by the underlying S3 client.
    """
    s3_client = get_s3_client()          # <-- enforces ExpectedBucketOwner
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,                     # <-- the required payload
            ContentType=content_type,
        )
        log.info(
            "Uploaded s3://%s/%s (%d bytes, type=%s)",
            bucket,
            key,
            len(data),
            content_type,
        )
    except Exception as exc:               # pragma: no cover – exercised via tests
        log.error("Failed to upload s3://%s/%s: %s", bucket, key, exc)
        raise


# =============================================================================
# 4️⃣  End‑to‑end signing + upload workflow
# =============================================================================
def sign_and_upload_to_s3(
    file_path: str,
    bucket: str,
    key: str,
    kms_key_id: str,
    content_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    """
    Complete workflow:
        1️⃣  Hash the file (SHA‑256)
        2️⃣  Sign the digest with AWS KMS
        3️⃣  Upload the raw file bytes to S3 (secure client)

    Args:
        file_path: Path to the file to sign and upload.
        bucket: Destination S3 bucket.
        key: Destination object key inside the bucket.
        kms_key_id: AWS KMS key ID / ARN used for signing.
        content_type: MIME type for the uploaded object.

    Returns:
        ``dict`` containing:
            * ``digest`` – hex SHA‑256 of the source file
            * ``signature`` – base64‑encoded KMS signature
            * ``s3_location`` – ``s3://bucket/key`` string

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ClientError: Propagated from KMS or S3 operations.
    """
    # -----------------------------------------------------------------
    # 1️⃣  Compute file hash
    # -----------------------------------------------------------------
    digest_hex = sha256_hex(file_path)
    log.info("Computed SHA‑256 for %s → %s", file_path, digest_hex)

    # -----------------------------------------------------------------
    # 2️⃣  Sign the digest with KMS
    # -----------------------------------------------------------------
    signature = kms_sign(kms_key_id, digest_hex)
    log.info("Obtained KMS signature (%d bytes)", len(signature))

    # -----------------------------------------------------------------
    # 3️⃣  Read the raw file bytes (to be uploaded)
    # -----------------------------------------------------------------
    with open(file_path, "rb") as f:
        raw_data = f.read()

    # -----------------------------------------------------------------
    # 4️⃣  Securely upload to S3
    # -----------------------------------------------------------------
    s3_put(bucket=bucket, key=key, data=raw_data, content_type=content_type)

    # -----------------------------------------------------------------
    # 5️⃣  Return a handy summary dict
    # -----------------------------------------------------------------
    return {
        "digest": digest_hex,
        "signature": base64.b64encode(signature).decode("utf-8"),
        "s3_location": f"s3://{bucket}/{key}",
    }
    
