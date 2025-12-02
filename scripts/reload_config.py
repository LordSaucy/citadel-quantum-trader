#!/usr/bin/env python3
import os, sys, requests

API = os.getenv("ADMIN_API", "https://admin.citadel.local/api/config/reload")
TOKEN = os.getenv("ADMIN_TOKEN")  # set this to the JWT or Vault token you use for the UI

if not TOKEN:
    sys.exit("Set ADMIN_TOKEN env var first")

resp = requests.post(API, headers={"Authorization": f"Bearer {TOKEN}"})
print(resp.status_code, resp.text)
