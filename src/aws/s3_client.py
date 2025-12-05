# src/aws/s3_client.py
import boto3
from boto3.session import Config
from .constants import EXPECTED_BUCKET_OWNER

def get_s3_client():
    """
    Return a boto3 S3 client that enforces bucket‑ownership verification.
    All CQT code should import and use this function instead of
    ``boto3.client('s3')`` directly.
    """
    return boto3.client(
        "s3",
        config=Config(signature_version="s3v4"),
        expected_bucket_owner=EXPECTED_BUCKET_OWNER,
    )
