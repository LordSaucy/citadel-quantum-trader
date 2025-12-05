import base64
import hashlib
import json
import os
from botocore.exceptions import ClientError
import boto3

def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client('s3')

def fetch_latest_snapshot():
    """
    Fetch the latest merkle root and signature from S3.
    
    Returns:
        tuple: (root_hex, signature_b64, snapshot_key)
        
    Raises:
        RuntimeError: If no ledger snapshot found
    """
    bucket = os.getenv('LEDGER_S3_BUCKET')
    if not bucket:
        raise ValueError("LEDGER_S3_BUCKET environment variable not set")
    
    s3 = get_s3_client()
    
    # List objects in S3 bucket
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix='ledger/')
    except ClientError as e:
        raise RuntimeError(f"Failed to list S3 objects: {e}")
    
    # Sort objects by key descending (newest first, assuming timestamp in key)
    objs = sorted(resp.get('Contents', []), key=lambda o: o['Key'], reverse=True)
    
    # Find latest merkle root and signature pair
    for obj in objs:
        if obj['Key'].endswith('_merkle_root.txt'):
            root_key = obj['Key']
            sig_key = root_key.replace('_merkle_root.txt', '_signature.txt')
            
            # Fetch merkle root from S3
            try:
                root_obj = s3.get_object(Bucket=bucket, Key=root_key)
                root = root_obj['Body'].read().decode('utf-8').strip()
            except ClientError as e:
                raise RuntimeError(f"Failed to fetch merkle root {root_key}: {e}")
            
            # Fetch signature from S3
            try:
                sig_obj = s3.get_object(Bucket=bucket, Key=sig_key)
                sig = sig_obj['Body'].read().decode('utf-8').strip()
            except ClientError as e:
                raise RuntimeError(f"Failed to fetch signature {sig_key}: {e}")
            
            return root, sig, root_key
    
    raise RuntimeError("No ledger snapshot found in S3")


def verify_signature(root_hex, sig_b64, kms_key_arn):
    """
    Verify a merkle root signature using AWS KMS.
    
    Arguments:
        root_hex: Merkle root as hex string
        sig_b64: Signature as base64-encoded string
        kms_key_arn: ARN of the KMS signing key
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    kms = boto3.client('kms')
    
    try:
        resp = kms.verify(
            KeyId=kms_key_arn,
            Message=bytes.fromhex(root_hex),
            MessageType='DIGEST',
            Signature=base64.b64decode(sig_b64),
            SigningAlgorithm='RSASSA_PKCS1_V1_5_SHA_256',
        )
        return resp['SignatureValid']
    except ClientError as e:
        raise RuntimeError(f"KMS verification failed: {e}")


def main():
    """Main entry point - verify latest ledger snapshot."""
    kms_key_arn = os.getenv('KMS_SIGNING_KEY_ARN')
    
    if not kms_key_arn:
        raise ValueError("KMS_SIGNING_KEY_ARN environment variable not set")
    
    # Fetch latest snapshot
    root, sig, key = fetch_latest_snapshot()
    
    # Verify signature
    ok = verify_signature(root, sig, kms_key_arn)
    
    # Print result
    status = 'VALID' if ok else 'INVALID'
    print(f"Snapshot {key} verification: {status}")
    
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}", file=__import__('sys').stderr)
        exit(1)
