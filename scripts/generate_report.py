#!/usr/bin/env python3
"""
Citadel Quantum Trader – Audit‑ready daily/weekly report generator.

Workflow:
    1️⃣ Pull the latest immutable ledger snapshot from S3.
    2️⃣ Load the current config.yaml.
    3️⃣ Load the most recent back‑test summary (JSON file <snapshot_name>_summary.json).
    4️⃣ Load Monte‑Carlo DD percentiles (<snapshot_name>_mc_dd.json).
    5️⃣ Render a PDF (ReportLab) that contains all of the above.
    6️⃣ Email the PDF to a list of recipients **or** post it to Slack.
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
import ssl
import smtplib
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import boto3
import pandas as pd
import yaml
from botocore.exceptions import BotoCoreError, ClientError
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle)

# ----------------------------------------------------------------------
# Configuration (environment variables – keep secrets out of the repo)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    # S3
    s3_bucket: str = os.getenv("CITADEL_REPORT_S3_BUCKET", "citadel-audit")
    s3_prefix: str = os.getenv("CITADEL_REPORT_S3_PREFIX", "snapshots/")

    # Email (SMTP over TLS)
    smtp_server: str = os.getenv("CITADEL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("CITADEL_SMTP_PORT", "465"))  # SSL
    smtp_user: str | None = os.getenv("CITADEL_SMTP_USER")      # e‑mail address
    smtp_password: str | None = os.getenv("CITADEL_SMTP_PASSWORD")
    email_recipients: List[str] = [
        addr.strip() for addr in os.getenv("CITADEL_REPORT_EMAILS", "").split(",") if addr.strip()
    ]

    # Slack (optional)
    slack_webhook_url: str | None = os.getenv("CITADEL_SLACK_WEBHOOK")

    # Misc
    report_frequency: str = os.getenv("CITADEL_REPORT_FREQ", "daily")  # used for naming only

    def __post_init__(self) -> None:
        # Validate required fields early – fail fast if mis‑configured
        missing = []
        if not self.smtp_user:
            missing.append("CITADEL_SMTP_USER")
        if not self.smtp_password:
            missing.append("CITADEL_SMTP_PASSWORD")
        if not self.email_recipients:
            missing.append("CITADEL_REPORT_EMAILS")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables for email delivery: {', '.join(missing)}"
            )


# ----------------------------------------------------------------------
# Logging – single source of truth
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cqt_report")

# ----------------------------------------------------------------------
# Helper – strong TLS context (TLS 1.2+)
# ----------------------------------------------------------------------
def _tls_context() -> ssl.SSLContext:
    """
    Return an SSLContext that enforces TLS 1.2 or newer.
    Python 3.10+ already defaults to the strongest protocol, but we
    explicitly set it for clarity and compatibility with older runtimes.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    # Enable only TLS 1.2 and TLS 1.3 (if available)
    ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_default_certs()
    return ctx


# ----------------------------------------------------------------------
# S3 utilities
# ----------------------------------------------------------------------
def _latest_s3_object(s3: boto3.client, bucket: str, prefix: str) -> Dict[str, Any]:
    """
    Return the metadata dict of the newest object under ``prefix``.
    Raises FileNotFoundError if the prefix is empty.
    """
    paginator = s3.get_paginator("list_objects_v2")
    latest: Dict[str, Any] | None = None

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if latest is None or obj["LastModified"] > latest["LastModified"]:
                latest = obj

    if latest is None:
        raise FileNotFoundError(f"No objects found in s3://{bucket}/{prefix}")

    return latest


def download_snapshot(s3: boto3.client) -> Dict[str, Any]:
    """
    Download the newest ledger snapshot from S3 and return a dict:

        {
            "snapshot": <parsed JSON>,
            "key":      <S3 object key>
        }

    The function raises ``RuntimeError`` on any AWS error.
    """
    try:
        obj = _latest_s3_object(s3, EnvConfig().s3_bucket, EnvConfig().s3_prefix)
        log.info("Downloading latest ledger snapshot: %s", obj["Key"])
        resp = s3.get_object(Bucket=EnvConfig().s3_bucket, Key=obj["Key"])
        snapshot = json.load(resp["Body"])
        return {"snapshot": snapshot, "key": obj["Key"]}
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 download failed: {exc}") from exc


def _load_json_sidecar(s3: boto3.client, snapshot_key: str, suffix: str) -> Dict[str, Any]:
    """
    Helper to load a JSON side‑car file that follows the naming convention:
        <snapshot_base>_<suffix>.json
    """
    base = snapshot_key.rsplit(".", 1)[0]  # strip .json
    sidecar_key = f"{base}{suffix}"
    log.info("Fetching side‑car %s", sidecar_key)
    try:
        resp = s3.get_object(Bucket=EnvConfig().s3_bucket, Key=sidecar_key)
        return json.load(resp["Body"])
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to load side‑car {sidecar_key}: {exc}") from exc


def load_backtest_summary(s3: boto3.client, snapshot_key: str) -> Dict[str, Any]:
    """Load <snapshot>_summary.json."""
    return _load_json_sidecar(s3, snapshot_key, "_summary.json")


def load_mc_dd(s3: boto3.client, snapshot_key: str) -> Dict[str, Any]:
    """Load <snapshot>_mc_dd.json."""
    return _load_json_sidecar(s3, snapshot_key, "_mc_dd.json")


# ----------------------------------------------------------------------
# Config loader (mounted at /opt/citadel/config/config.yaml)
# ----------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    cfg_path = Path("/opt/citadel/config/config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded config.yaml (version %s)", cfg.get("version", "N/A"))
    return cfg


# ----------------------------------------------------------------------
# PDF rendering (ReportLab)
# ----------------------------------------------------------------------
def render_pdf(
    snapshot: Dict[str, Any],
    config: Dict[str, Any],
    summary: Dict[str, Any],
    mc_dd: Dict[str, Any],
    out_path: Path,
) -> None:
    """
    Build a one‑page (or multi‑page) PDF report and write it to ``out_path``.
    """
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=18,
    )
    styles = getSampleStyleSheet()
    elems: List[Any] = []

    # ---- Title ----------------------------------------------------
    title = f"Citadel Quantum Trader – {EnvConfig().report_frequency.capitalize()} Report"
    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Spacer(1, 12))

    # ---- Generation timestamp --------------------------------------
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    elems.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    # ---- Config summary -------------------------------------------
    elems.append(Paragraph("<b>Live Configuration (config.yaml)</b>", styles["Heading2"]))
    cfg_table = [[k, str(v)] for k, v in config.items()]
    tbl = Table(cfg_table, colWidths=[150, 350])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    # ---- Back‑test summary ----------------------------------------
    elems.append(Paragraph("<b>Back‑test Summary</b>", styles["Heading2"]))
    bt_data = [
        ["Metric", "Value"],
        ["Total Trades", summary.get("total_trades")],
        ["Win Rate (%)", f"{summary.get('win_rate', 0):.2f}"],
        ["Expectancy", f"{summary.get('expectancy', 0):.4f}"],
        ["Profit Factor", f"{summary.get('profit_factor', 0):.2f}"],
        ["Max Draw‑down (%)", f"{summary.get('max_drawdown', 0):.2f}"],
        ["Total Gross P&L", f"${summary.get('gross_total_profit', 0):,.2f}"],
        ["Total Net P&L", f"${summary.get('net_total_profit', 0):,.2f}"],
        ["Avg Cost / Trade", f"${summary.get('average_cost_per_trade', 0):,.2f}"],
    ]
    bt_tbl = Table(bt_data, colWidths=[200, 300])
    bt_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elems.append(bt_tbl)
    elems.append(Spacer(1, 12))

    # ---- Monte‑Carlo DD percentiles -------------------------------
    elems.append(Paragraph("<b>Monte‑Carlo Draw‑down Percentiles</b>", styles["Heading2"]))
    perc_data = [["Percentile", "Draw‑down (%)"]]
    for pct, dd in sorted(mc_dd.items(), key=lambda x: float(x[0].strip("%"))):
        perc_data.append([f"{pct}%", f"{dd:.2f}"])
    perc_tbl = Table(perc_data, colWidths=[200, 300])
    perc_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elems.append(perc_tbl)
    elems.append(Spacer(1, 12))

    # ---- Immutable ledger snapshot quick stats --------------------
    elems.append(Paragraph("<b>Immutable Ledger Snapshot</b>", styles["Heading2"]))
    ledger_stats = [
        ["Trades in snapshot", len(snapshot.get("trades", []))],
        ["Snapshot timestamp", snapshot.get("timestamp", "N/A")],
    ]
    led_tbl = Table(ledger_stats, colWidths=[200, 300])
    led_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elems.append(led_tbl)

    # Build the PDF
    doc.build(elems)
    log.info("PDF report written to %s", out_path)


# ----------------------------------------------------------------------
# Email delivery (SMTP over TLS)
# ----------------------------------------------------------------------
def send_email(pdf_path: Path) -> None:
    cfg = EnvConfig()
    if not cfg.smtp_user or not cfg.smtp_password or not cfg.email_recipients:
        log.warning("SMTP credentials or recipient list missing – skipping e‑mail")
        return

    msg = EmailMessage()
    msg["Subject"] = f"Citadel {cfg.report_frequency.capitalize()} Report – {datetime.utcnow().date()}"
    msg["From"] = cfg.smtp_user
    msg["To"] = ", ".join(cfg.email_recipients)
    msg.set_content(
        f"Attached is the latest {cfg.report_frequency} audit‑ready report for the Citadel Quantum Trader.\n\n"
        f"Generated on {datetime.utcnow().isoformat()} UTC."
    )

    # Attach PDF
    with pdf_path.open("rb") as f:
        pdf_bytes = f.read()
    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=pdf_path.name,
    )

    ctx = _tls_context()
    log.info("Sending report e‑mail to %s", cfg.email_recipients)
    with smtplib.SMTP_SSL(cfg.smtp_server, cfg.smtp_port, context=ctx) as server:
        server.login(cfg.smtp_user, cfg.smtp_password)
        server.send_message(msg)
    log.info("E‑mail sent successfully")


# ----------------------------------------------------------------------
# Slack delivery (incoming webhook)
# ----------------------------------------------------------------------
def post_to_slack(pdf_path: Path) -> None:
    cfg = EnvConfig()
    if not cfg.slack_webhook_url:
        log.warning("Slack webhook URL not set – skipping Slack post")
        return

    import requests

    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        payload = {
            "filename": pdf_path.name,
            "channels": "#citadel-reports",
            "title": f"Citadel {cfg.report_frequency.capitalize()} Report",
            "initial_comment": f"Here is the latest {cfg.report_frequency} audit‑ready report.",
        }

    log.info("Posting report to Slack")
    response = requests.post(cfg.slack_webhook_url, data=payload, files=files)

    if response.status_code != 200:
        log.error("Slack post failed: %s – %s", response.status_code, response.text)
        raise RuntimeError(f"Slack post failed: {response.text}")
    log.info("Report posted to Slack successfully")


# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main() -> None:
    """
    Orchestrates the whole workflow:
        * download snapshot & side‑cars from S3
        * load config.yaml
        * render PDF
        * deliver via e‑mail and/or Slack
    """
    # Verify we are running on a supported Python version (>=3.10)
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10+ required – current version is {sys.version}"
        )

    cfg = EnvConfig()  # validates env vars early
    s3 = boto3.client("s3")

    try:
        # 1️⃣  Snapshot + key
        data = download_snapshot(s3)
        snapshot = data["snapshot"]
        snapshot_key = data["key"]

        # 2️⃣  Config
        config = load_config()

        # 3️⃣  Back‑test summary
        summary = load_backtest_summary(s3, snapshot_key

      # 4️⃣  Monte‑Carlo DD percentiles
        mc_dd = load_mc_dd(s3, snapshot_key)

        # 5️⃣  Render the PDF into a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"citadel_report_{cfg.report_frequency}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
            render_pdf(snapshot, config, summary, mc_dd, pdf_path)

            # 6️⃣  Deliver the report
            try:
                send_email(pdf_path)
            except Exception as exc:
                log.error("E‑mail delivery failed: %s", exc)

            try:
                post_to_slack(pdf_path)
            except Exception as exc:
                log.error("Slack delivery failed: %s", exc)

        log.info("Report generation completed successfully")

    except Exception as exc:
        # Top‑level catch – ensures we exit with a non‑zero status for CI pipelines
        log.exception("Report generation aborted due to an unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
