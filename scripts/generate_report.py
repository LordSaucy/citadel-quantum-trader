#!/usr/bin/env python3
"""
Citadel Quantum Trader – Audit‑ready daily/weekly report generator.

Workflow:
1️⃣  Pull the latest immutable ledger snapshot from S3.
2️⃣  Load the current config.yaml.
3️⃣  Load the most recent back‑test summary (stored alongside the snapshot
   – we assume a JSON file named <snapshot_name>_summary.json exists).
4️⃣  Load Monte‑Carlo DD percentiles (JSON file <snapshot_name>_mc_dd.json).
5️⃣  Render a PDF (ReportLab) that contains all of the above.
6️⃣  Email the PDF to a list of recipients **or** post it to Slack.
"""

import os
import json
import logging
import smtplib
import ssl
from datetime import datetime
from pathlib import Path
from email.message import EmailMessage

import boto3
import yaml
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# ----------------------------------------------------------------------
# Configuration – read from environment (keeps secrets out of the repo)
# ----------------------------------------------------------------------
S3_BUCKET          = os.getenv('CITADEL_REPORT_S3_BUCKET', 'citadel-audit')
S3_PREFIX          = os.getenv('CITADEL_REPORT_S3_PREFIX', 'snapshots/')   # optional folder
SMTP_SERVER        = os.getenv('CITADEL_SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT          = int(os.getenv('CITADEL_SMTP_PORT', '465'))  # SSL
SMTP_USER          = os.getenv('CITADEL_SMTP_USER')            # e‑mail address
SMTP_PASSWORD      = os.getenv('CITADEL_SMTP_PASSWORD')
EMAIL_RECIPIENTS   = os.getenv('CITADEL_REPORT_EMAILS', '').split(',')  # comma‑separated list
SLACK_WEBHOOK_URL  = os.getenv('CITADEL_SLACK_WEBHOOK')          # optional
REPORT_FREQUENCY   = os.getenv('CITADEL_REPORT_FREQ', 'daily')   # just for naming

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
log = logging.getLogger('cqt_report')

# ----------------------------------------------------------------------
# Helper – download the *latest* object that matches a prefix
# ----------------------------------------------------------------------
def _latest_s3_object(s3_client, bucket: str, prefix: str) -> dict:
    """Return the S3 object dict (Key, LastModified, ...) for the newest file."""
    paginator = s3_client.get_paginator('list_objects_v2')
    latest = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if latest is None or obj['LastModified'] > latest['LastModified']:
                latest = obj
    if not latest:
        raise FileNotFoundError(f"No objects found in s3://{bucket}/{prefix}")
    return latest

# ----------------------------------------------------------------------
# 1️⃣ Pull ledger snapshot (JSON) from S3
# ----------------------------------------------------------------------
def _download_snapshot(s3_client) -> dict:
    obj = _latest_s3_object(s3_client, S3_BUCKET, S3_PREFIX)
    log.info("Downloading latest ledger snapshot: %s", obj['Key'])
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
    snapshot = json.load(resp['Body'])
    return snapshot, obj['Key']

# ----------------------------------------------------------------------
# 2️⃣ Load config.yaml (mounted in the container at /opt/citadel/config)
# ----------------------------------------------------------------------
def _load_config() -> dict:
    cfg_path = Path('/opt/citadel/config/config.yaml')
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded config.yaml (version %s)", cfg.get('version', 'N/A'))
    return cfg

# ----------------------------------------------------------------------
# 3️⃣ Load back‑test summary (assumed to be stored next to the snapshot)
# ----------------------------------------------------------------------
def _load_backtest_summary(s3_client, snapshot_key: str) -> dict:
    # Convention: snapshot file = ledger_snapshot_YYYYMMDD_HH.json
    # Summary file   = ledger_snapshot_YYYYMMDD_HH_summary.json
    base = snapshot_key.rsplit('.', 1)[0]          # strip .json
    summary_key = f"{base}_summary.json"
    log.info("Fetching back‑test summary: %s", summary_key)
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=summary_key)
    return json.load(resp['Body'])

# ----------------------------------------------------------------------
# 4️⃣ Load Monte‑Carlo DD percentiles (same naming convention)
# ----------------------------------------------------------------------
def _load_mc_dd(s3_client, snapshot_key: str) -> dict:
    base = snapshot_key.rsplit('.', 1)[0]
    mc_key = f"{base}_mc_dd.json"
    log.info("Fetching Monte‑Carlo DD percentiles: %s", mc_key)
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=mc_key)
    return json.load(resp['Body'])

# ----------------------------------------------------------------------
# 5️⃣ Render PDF with ReportLab
# ----------------------------------------------------------------------
def _render_pdf(
    snapshot: dict,
    config: dict,
    summary: dict,
    mc_dd: dict,
    out_path: Path
) -> None:
    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER,
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []

    # ---- Title ----------------------------------------------------
    title = f"Citadel Quantum Trader – {REPORT_FREQUENCY.capitalize()} Report"
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    # ---- Timestamp ------------------------------------------------
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    elements.append(Paragraph(f"Generated: {now}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # ---- Config summary --------------------------------------------
    elements.append(Paragraph("<b>Live Configuration (config.yaml)</b>", styles['Heading2']))
    cfg_table_data = [[k, str(v)] for k, v in config.items()]
    cfg_tbl = Table(cfg_table_data, colWidths=[150, 350])
    cfg_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(cfg_tbl)
    elements.append(Spacer(1, 12))

    # ---- Back‑test summary -----------------------------------------
    elements.append(Paragraph("<b>Back‑test Summary</b>", styles['Heading2']))
    bt_data = [
        ['Metric', 'Value'],
        ['Total Trades', summary.get('total_trades')],
        ['Win Rate (%)', f"{summary.get('win_rate',0):.2f}"],
        ['Expectancy', f"{summary.get('expectancy',0):.4f}"],
        ['Profit Factor', f"{summary.get('profit_factor',0):.2f}"],
        ['Max Draw‑down (%)', f"{summary.get('max_drawdown',0):.2f}"],
        ['Total Gross P&L', f"${summary.get('gross_total_profit',0):,.2f}"],
        ['Total Net P&L', f"${summary.get('net_total_profit',0):,.2f}"],
        ['Average Cost / Trade', f"${summary.get('average_cost_per_trade',0):,.2f}"]
    ]
    bt_tbl = Table(bt_data, colWidths=[200, 300])
    bt_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(bt_tbl)
    elements.append(Spacer(1, 12))

    # ---- Monte‑Carlo DD percentiles --------------------------------
    elements.append(Paragraph("<b>Monte‑Carlo Draw‑down Percentiles</b>", styles['Heading2']))
    perc_data = [['Percentile', 'Draw‑down (%)']]
    for pct, dd in sorted(mc_dd.items()):
        perc_data.append([f"{pct}%", f"{dd:.2f}"])
    perc_tbl = Table(perc_data, colWidths=[200, 300])
    perc_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(perc_tbl)
    elements.append(Spacer(1, 12))

    # ---- Ledger snapshot stats (optional quick glance) -------------
    # Example: total number of trades stored in the immutable ledger
    elements.append(Paragraph("<b>Immutable Ledger Snapshot</b>", styles['Heading2']))
    ledger_stats = [
        ['Trades in snapshot', len(snapshot.get('trades', []))],
        ['Snapshot timestamp', snapshot.get('timestamp', 'N/A')]
    ]
    led_tbl = Table(ledger_stats, colWidths=[200, 300])
    led_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(led_tbl)

    # Build the PDF
    doc.build(elements)
    log.info("PDF report written to %s", out_path)

# ----------------------------------------------------------------------
# 6️⃣ Delivery – e‑mail (SMTP) or Slack webhook
# ----------------------------------------------------------------------
def _send_email(pdf_path: Path) -> None:
    if not SMTP_USER or not SMTP_PASSWORD or not EMAIL_RECIPIENTS:
        log.warning("SMTP credentials or recipient list missing – skipping e‑mail")
        return

    msg = EmailMessage()
    msg['Subject'] = f"Citadel {REPORT_FREQUENCY.capitalize()} Report – {datetime.utcnow().date()}"
    msg['From'] = SMTP_USER
    msg['To'] = ', '.join(EMAIL_RECIPIENTS)
    msg.set_content(f"Attached is the latest {REPORT_FREQUENCY} audit‑ready report for the Citadel Quantum Trader.\n\n"
                    f"Generated on {datetime.utcnow().isoformat()} UTC.")

    # Attach PDF
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    msg.add_attachment(pdf_data,
                       maintype='application',
                       subtype='pdf',
                       filename=pdf_path.name)

    context = ssl.create_default_context()
    log.info("Sending report e‑mail to %s", EMAIL_RECIPIENTS)
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
    log.info("E‑mail sent successfully")

def _post_to_slack(pdf_path: Path) -> None:
    if not SLACK_WEBHOOK_URL:
        log.warning("Slack webhook URL not set – skipping Slack post")
        return

    import requests
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        payload = {
            'filename': pdf_path.name,
            'channels': '#citadel-reports',   # you can also make this env‑var
            'title': f"Citadel {REPORT_FREQUENCY.capitalize()} Report",
            'initial_comment': f"Here is the latest {REPORT_FREQUENCY} audit‑ready report."
        }
        log.info("Posting report to Slack")
        response = requests.post(
            SLACK_WEBHOOK_URL,
            data=payload,
            files=files
        )
    if response.status_code != 200:
        log.error("Slack post failed: %s – %s", response.status_code, response.text)
    else:
        log.info("Report posted to Slack successfully")

# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main() -> None:
    s3 = boto3.client('s3')
    try:
        snapshot, snapshot_key = _download_snapshot(s3)
        config = _load_config()
        summary = _load_backtest_summary(s3, snapshot_key)
        mc_dd = _load_mc_dd(s3, snapshot_key)

        # Temporary file for the PDF
        out_dir = Path('/tmp')
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"citadel_report_{REPORT_FREQUENCY}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"

        _render_pdf(snapshot, config, summary, mc_dd, pdf_path)

        # Delivery
        _send_email(pdf_path)
        _post_to_slack(pdf_path)

    except Exception as exc:
        log.exception("Report generation failed: %s", exc)
        raise

if __name__ == '__main__':
    main()
