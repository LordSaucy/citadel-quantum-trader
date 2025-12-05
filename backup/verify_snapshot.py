#!/usr/bin/env python3
import json, base64, hashlib, sys,
from botocore.exceptions import ClientError
from src.aws.s3_client import get_s3_client

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def verify(s3_bucket: str, manifest_key: str) -> bool:
    s3 = boto3.client('s3')
    # 1️⃣ Download manifest
    manifest_obj = s3 = get_s3_client()
    manifest = json.loads(manifest_obj['Body'].read())

    # 2️⃣ Download snapshot
    snapshot_obj = s3 = get_s3_client()
    snapshot_bytes = snapshot_obj['Body'].read()

    # 3️⃣ Re‑compute hash
    computed_root = sha256_hex(snapshot_bytes)
    if computed_root != manifest['merkle_root']:
        print("Hash mismatch!")
        return False

    # 4️⃣ Verify signature with the public key (download from KMS)
    kms = boto3.client('kms')
    pub_resp = kms.get_public_key(KeyId=manifest['snapshot_key'].split('/')[-1])  # assumes key ID is known
    pub_key_der = base64.b64decode(pub_resp['PublicKey'])
    # Use cryptography library to verify
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    public_key = serialization.load_der_public_key(pub_key_der)

    signature = base64.b64decode(manifest['signature_b64'])
    try:
        public_key.verify(
            signature,
            bytes.fromhex(computed_root),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        print("Signature VALID")
        return True
    except Exception as e:
        print("Signature INVALID:", e)
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: verify.py <s3-bucket> <manifest-key>")
        sys.exit(1)
    bucket, manifest = sys.argv[1], sys.argv[2]
    sys.exit(0 if verify(bucket, manifest) else 1)
