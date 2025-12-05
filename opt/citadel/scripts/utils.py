import json
import hashlib
import base64
import logging
import os
import boto3
from botocore.exceptions import ClientError
from src.aws.s3_client import get_s3_client

log = logging.getLogger('cqt.signing')


def sha256_hex(file_path: str) -> str:
    """
    Return the SHA‑256 hex digest of a file.
    
    Args:
        file_path: Path to file to hash
        
    Returns:
        Hex string of SHA-256 digest
        
    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def kms_sign(key_id: str, digest_hex: str) -> bytes:
    """
    Ask AWS KMS (or CloudHSM via KMS) to sign a SHA‑256 digest.
    
    Args:
        key_id: AWS KMS key ID or ARN
        digest_hex: SHA-256 digest as hex string
        
    Returns:
        Raw signature bytes
        
    Raises:
        ClientError: If KMS signing fails
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


def s3_put(bucket: str, key: str, content_type: str = 'application/octet-stream') -> None:
    """
    Upload a binary blob to S3 with server‑side encryption (SSE‑KMS).
    
    Uses the S3 client from get_s3_client() which is pre-configured with
    SSE-KMS encryption settings.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path)
        content_type: MIME type of object (default: application/octet-stream)
        
    Returns:
        None
        
    Raises:
        ClientError: If S3 upload fails
    """
    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            ContentType=content_type,
            ServerSideEncryption='aws:kms'
        )
        log.info("Uploaded %s to s3://%s/%s", content_type, bucket, key)
    except ClientError as exc:
        log.error("S3 upload failed: %s", exc)
        raise


# ============================================================================
# Optional: Signing Workflow Helper
# ============================================================================

def sign_and_upload_to_s3(
    file_path: str,
    bucket: str,
    key: str,
    kms_key_id: str,
    content_type: str = 'application/octet-stream'
) -> dict:
    """
    Complete workflow: hash file, sign digest with KMS, upload to S3.
    
    Args:
        file_path: Path to file to sign and upload
        bucket: S3 bucket name
        key: S3 object key (path)
        kms_key_id: AWS KMS key ID/ARN for signing
        content_type: MIME type (default: application/octet-stream)
        
    Returns:
        Dictionary with 'digest', 'signature', 's3_location'
        
    Raises:
        FileNotFoundError: If file_path does not exist
        ClientError: If KMS or S3 operations fail
    """
    # 1. Hash the file
    digest_hex = sha256_hex(file_path)
    log.info("File hash: %s", digest_hex)
    
    # 2. Sign the digest with KMS
    signature = kms_sign(kms_key_id, digest_hex)
    log.info("KMS signature generated: %d bytes", len(signature))
    
    # 3. Upload to S3
    s3_put(bucket, key, content_type)
    
    return {
        'digest': digest_hex,
        'signature': base64.b64encode(signature).decode('utf-8'),
        's3_location': f's3://{bucket}/{key}'
    }
