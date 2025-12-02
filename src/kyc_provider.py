# kyc_provider.py
import os
import json
import hashlib
import requests
from typing import Dict, Any

# ------------------------------------------------------------------
# Configuration – read from environment (populated by Vault or ECS task role)
# ------------------------------------------------------------------
PERSONA_API_URL   = os.getenv('PERSONA_API_URL', 'https://withpersona.com/api/v1')
PERSONA_API_TOKEN = os.getenv('PERSONA_API_TOKEN')          # Vault‑injected
JUMIO_API_URL     = os.getenv('JUMIO_API_URL', 'https://api.jumio.com')
JUMIO_API_TOKEN   = os.getenv('JUMIO_API_TOKEN')          # Vault‑injected

# ------------------------------------------------------------------
# Helper – store a **hash** of the verification token, not the raw token
# ------------------------------------------------------------------
def _hash_token(token: str) -> str:
    """Return a SHA‑256 hex digest of the raw verification token."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------
# Persona implementation
# ------------------------------------------------------------------
def _persona_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Persona's /verification endpoint.
    Payload must contain the fields required by your Persona workflow
    (e.g., name, dob, address, document_image).
    Returns a dict with at least: { "status": "verified", "token": "<raw>" }
    """
    url = f'{PERSONA_API_URL}/verification'
    headers = {
        'Authorization': f'Bearer {PERSONA_API_TOKEN}',
        'Content-Type': 'application/json',
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # Expected shape: {"status":"verified","verificationToken":"abc123..."}
    return {
        'status': data.get('status'),
        'raw_token': data.get('verificationToken')
    }


# ------------------------------------------------------------------
# Jumio implementation
# ------------------------------------------------------------------
def _jumio_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Jumio's /scan endpoint.
    Payload must contain the fields required by your Jumio workflow.
    Returns a dict with at least: { "status": "APPROVED", "token": "<raw>" }
    """
    url = f'{JUMIO_API_URL}/scan'
    headers = {
        'Authorization': f'Bearer {JUMIO_API_TOKEN}',
        'Content-Type': 'application/json',
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        'status': data.get('status'),
        'raw_token': data.get('verificationToken')
    }


# ------------------------------------------------------------------
# Public façade – choose provider via env‑var or explicit argument
# ------------------------------------------------------------------
def verify_identity(payload: Dict[str, Any], provider: str = 'persona') -> Dict[str, Any]:
    """
    Unified entry point.
    Returns a dict:
        {
            "provider": "persona" | "jumio",
            "status":   "verified" | "APPROVED" | "FAILED",
            "token_hash": "<sha256>",
            "raw_token":  "<only stored temporarily>"
        }
    The caller should **immediately discard** `raw_token` after hashing.
    """
    if provider.lower() == 'persona':
        result = _persona_verify(payload)
    elif provider.lower() == 'jumio':
        result = _jumio_verify(payload)
    else:
        raise ValueError(f'Unsupported KYC provider: {provider}')

    token_hash = _hash_token(result['raw_token']) if result.get('raw_token') else None

    return {
        'provider'   : provider.lower(),
        'status'     : result.get('status'),
        'token_hash' : token_hash,
        'raw_token'  : result.get('raw_token')   # keep only in memory; do NOT persist
    }
