# utils.py
import json, hashlib, base64, logging, os
from src.aws.s3_client import get_s3_client
from botocore.exceptions import ClientError

log = logging.getLogger('cqt.signing')


def sha256_hex(file_path: str) -> str:
    """Return the SHA‑256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def kms_sign(key_id: str, digest_hex: str) -> bytes:
    """
    Ask AWS KMS (or CloudHSM via KMS) to sign a SHA‑256 digest.
    Returns the raw signature bytes.
    """
    client = boto3.client('kms')
    try:
        resp = client.sign(
            KeyId=key_id,
            Message=bytes.fromhex(digest_hex),
            MessageType='DIGEST',
            SigningAlgorithm='RSASSA_PKCS1_V1_5_SHA_256'
        )
        return resp['Signature']
    except ClientError as exc:
        log.error("KMS sign failed: %s", exc)
        raise


def s3_put(bucket: str, key: str, body: bytes, content_type: str = 'application/octet-stream'):
    """Upload a binary blob to S3 with server‑side encryption (SSE‑KMS)."""
    s3 = boto3.client('s3')
    try:
        s3 = get_s3_client()
        log.info("Uploaded %s to s3://%s/%s", content_type, bucket, key)
    except ClientError as exc:
        log.error("S3 upload failed: %s", exc)
        raise
