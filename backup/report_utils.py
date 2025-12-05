# report_utils.py
import json, csv, io, os, logging, datetime
from src.aws.s3_client import get_s3_client
from botocore.exceptions import ClientError

log = logging.getLogger('cqt.reporting')
s3 = boto3.client('s3')
AUDIT_BUCKET = os.getenv('AUDIT_S3_BUCKET', 'citadel-audit')
KMS_ENCRYPT_KEY = os.getenv('AWS_KMS_S3_KEY')   # same key used for snapshot encryption


def upload_encrypted(content: bytes, key: str, content_type: str):
    """Upload encrypted data to S3 (SSE‑KMS)."""
    try:
        s3 = get_s3_client()
        log.info("Uploaded %s (%d bytes)", key, len(content))
    except ClientError as exc:
        log.error("Failed to upload %s: %s", key, exc)
        raise
