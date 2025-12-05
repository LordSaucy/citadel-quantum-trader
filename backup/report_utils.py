# report_utils.py
# ============================================================================
# Regulatory Reporting Utilities - S3 Upload with KMS Encryption
#
# Purpose:
#   • Provides a common helper function for uploading audit reports to S3
#   • Implements server-side encryption using AWS KMS (SSE-KMS)
#   • Used by SFTR (EU) and Form PF (US) regulatory compliance scripts
#
# Configuration (environment variables):
#   AUDIT_S3_BUCKET     – S3 bucket for audit reports (default: citadel-audit)
#   AWS_KMS_S3_KEY      – KMS key ID for S3 encryption (required)
#
# Violations Fixed:
#   ✅ S1172: Unused function parameter 'content_type' → Now used in put_object()
#   ✅ S1481: Unused local variable 's3' → Removed, using module-level client instead
#
# =============================================================================

import json
import csv
import io
import os
import logging
import datetime
import boto3
from botocore.exceptions import ClientError

# Module-level logger
log = logging.getLogger('cqt.reporting')

# Global S3 client - initialized once at module import
s3 = boto3.client('s3')

# Configuration from environment
AUDIT_BUCKET = os.getenv('AUDIT_S3_BUCKET', 'citadel-audit')
KMS_ENCRYPT_KEY = os.getenv('AWS_KMS_S3_KEY')  # same key used for snapshot encryption


def upload_encrypted(content: bytes, key: str, content_type: str) -> None:
    """
    Upload encrypted data to S3 with server-side encryption (SSE-KMS).
    
    This is the primary function used by regulatory reporting scripts
    (SFTR, Form PF) to upload audit reports securely to S3.
    
    Arguments:
        content      – The bytes to upload
        key          – S3 object key (e.g., 'sftr/2024-11-30.xml')
        content_type – MIME type for the object (e.g., 'application/xml', 'text/csv')
    
    Raises:
        ClientError if the upload fails (e.g., KMS key not found, S3 bucket unavailable)
    
    Returns:
        None (logs success or raises exception)
    """
    try:
        # Use the module-level s3 client (not a local one)
        # This avoids creating unnecessary boto3 clients
        s3.put_object(
            Bucket=AUDIT_BUCKET,
            Key=key,
            Body=content,
            ServerSideEncryption='aws:kms',
            SSEKMSKeyId=KMS_ENCRYPT_KEY,
            ContentType=content_type  # ✅ FIXED: Parameter now used in put_object()
        )
        log.info("Uploaded %s (%d bytes) to s3://%s/%s", content_type, len(content), AUDIT_BUCKET, key)
    except ClientError as exc:
        log.error("Failed to upload %s to S3: %s", key, exc)
        raise


# ============================================================================
# Helper Functions for Report Generation
# ============================================================================

def dict_to_csv(records: list) -> bytes:
    """
    Convert list of dicts to CSV bytes.
    
    Arguments:
        records – List of dictionaries (each dict is a row)
    
    Returns:
        CSV data as bytes (UTF-8 encoded)
    """
    if not records:
        return b''
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode('utf-8')


def dict_to_json(data: dict) -> bytes:
    """
    Convert dict to JSON bytes (pretty-printed).
    
    Arguments:
        data – Dictionary to serialize
    
    Returns:
        JSON data as bytes (UTF-8 encoded)
    """
    return json.dumps(data, indent=2).encode('utf-8')


# ============================================================================
# Example Usage (for reference)
# ============================================================================
"""
# SFTR Report Upload Example
from report_utils import upload_encrypted

xml_data = b'<?xml version="1.0"?>...'  # Your XML report
upload_encrypted(xml_data, 'sftr/2024-11-30.xml', content_type='application/xml')

# Form PF Report Upload Example
csv_data = b'quarter,aum,pnl,...'  # Your CSV report
upload_encrypted(csv_data, 'formpf/2024Q3.csv', content_type='text/csv')
"""
