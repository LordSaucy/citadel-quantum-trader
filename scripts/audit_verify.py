import boto3, base64, hashlib, json, os
from botocore.exceptions import ClientError

def fetch_latest_snapshot(bucket, prefix='ledger/'):
    s3 = boto3.client('s3')
    # List objects sorted descending by key (timestamp in key)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objs = sorted(resp.get('Contents', []), key=lambda o: o['Key'], reverse=True)
    # Expect pairs: *_merkle_root.txt and *_signature.txt
    for obj in objs:
        if obj['Key'].endswith('_merkle_root.txt'):
            root_key = obj['Key']
            sig_key = root_key.replace('_merkle_root.txt', '_signature.txt')
            # fetch both
            root = s3.get_object(Bucket=bucket, Key=root_key)['Body'].read().decode()
            sig  = s3.get_object(Bucket=bucket, Key=sig_key)['Body'].read().decode()
            return root, sig, root_key
    raise RuntimeError("No ledger snapshot found")

def verify_signature(root_hex, sig_b64, kms_key_arn):
    kms = boto3.client('kms')
    resp = kms.verify(
        KeyId=kms_key_arn,
        Message=bytes.fromhex(root_hex),
        MessageType='DIGEST',
        Signature=base64.b64decode(sig_b64),
        SigningAlgorithm='RSASSA_PKCS1_V1_5_SHA_256',
    )
    return resp['SignatureValid']

if __name__ == '__main__':
    bucket = os.getenv('LEDGER_S3_BUCKET')
    key_arn = os.getenv('KMS_SIGNING_KEY_ARN')
    root, sig, key = fetch_latest_snapshot(bucket)
    ok = verify_signature(root, sig, key_arn)
    print(f"Snapshot {key} verification: {'VALID' if ok else 'INVALID'}")
