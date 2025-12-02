from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import jwt, JWTError
import os

# -------------------------------------------------
# 1️⃣ Choose one:
#   • Simple JWT signed with a secret (good for internal use)
#   • OAuth2AuthorizationCodeBearer for Okta/Azure AD (recommended)
# -------------------------------------------------
SECRET_KEY = os.getenv("ADMIN_JWT_SECRET", "CHANGE_ME")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://YOUR_OKTA_DOMAIN/oauth2/v1/authorize",
    tokenUrl="https://YOUR_OKTA_DOMAIN/oauth2/v1/token",
    scopes={"admin": "Admin access"},
)


def decode_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from exc


def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_jwt(token)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges",
        )
    return payload
