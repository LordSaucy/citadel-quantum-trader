# src/alerting.py (or wherever send_alert lives)
import os
import requests

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
SLACK_TOKEN   = os.getenv("SLACK_BOT_TOKEN")   # bot token with files:write scope

def send_alert(title: str, severity: str, details: dict, image_path: str | None = None):
    # 1️⃣ Send the normal webhook (fallback for non‑image channels)
    payload = {
        "text": f"*{title}* – severity: {severity}",
        "attachments": [
            {"fields": [{"title": k, "value": str(v), "short": True} for k, v in details.items()]}
        ],
    }
    requests.post(SLACK_WEBHOOK, json=payload)

    # 2️⃣ If an image is supplied, upload it as a file
    if image_path and SLACK_TOKEN:
        with open(image_path, "rb") as fp:
            files = {"file": fp}
            data = {
                "channels": "#trading-alerts",
                "initial_comment": f"{title} – see attached heat‑map",
                "title": os.path.basename(image_path),
            }
            headers = {"Authorization": f"Bearer {SLACK_TOKEN}"}
            requests.post("https://slack.com/api/files.upload", data=data,
                          files=files, headers=headers)
