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


def s3_put(bucket: str, key: str, data: bytes, content_type: str = 'application/octet-stream') -> None:
    """
    Upload object to S3 bucket.
    
    Arguments:
        bucket: S3 bucket name
        key: Object key/path in bucket
        data: Raw bytes to upload (3rd positional parameter - NOT named 'body')
        content_type: MIME type (default: application/octet-stream)
    
    Returns:
        None
    
    Raises:
        Exception: On S3 upload failure
    """
    import boto3
    import logging
    
    s3_client = boto3.client('s3')
    
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type
        )
        logging.info(f"Uploaded s3://{bucket}/{key} ({len(data)} bytes, type={content_type})")
    except Exception as exc:
        logging.error(f"Failed to upload s3://{bucket}/{key}: {exc}")
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
    
