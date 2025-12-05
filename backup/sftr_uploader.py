#!/usr/bin/env python3
"""
SFTR (EU) – daily transaction report.
The format is a simplified XML that complies with the ESMA schema.
Only the fields required for a “transaction‑level” report are populated.
"""

import os, datetime, xml.etree.ElementTree as ET
import psycopg2
from report_utils import upload_encrypted
from datetime import datetime, timezone

DB_DSN = os.getenv('POSTGRES_DSN')   # e.g. "dbname=citadel user=citadel password=… host=postgres"
TODAY = datetime.date.today().isoformat()

def fetch_transactions():
    """Return a list of dicts for yesterday's trades."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute("""
        SELECT bucket_id, symbol, direction, volume, entry_price,
               sl, tp, pnl, timestamp
        FROM trades
        WHERE timestamp::date = %s
    """, (TODAY,))
    cols = [desc[0] for desc in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows

def build_sftr_xml(trades):
    root = ET.Element('SFTRReport', attrib={'creationDateTime': datetime.datetime.utcnow(datetime.timezone.utc).isoformat()})
    for tr in trades:
        tx = ET.SubElement(root, 'Transaction')
        ET.SubElement(tx, 'BucketID').text = str(tr['bucket_id'])
        ET.SubElement(tx, 'Instrument').text = tr['symbol']
        ET.SubElement(tx, 'Direction').text = tr['direction']
        ET.SubElement(tx, 'Quantity').text = str(tr['volume'])
        ET.SubElement(tx, 'Price').text = f"{tr['entry_price']:.5f}"
        ET.SubElement(tx, 'PnL').text = f"{tr['pnl']:.5f}"
        ET.SubElement(tx, 'Timestamp').text = tr['timestamp'].isoformat()
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)

def main():
    trades = fetch_transactions()
   

def main():
    trades = fetch_transactions()
    if not trades:
        log.info("No trades for %s – nothing to report.", TODAY)
        return

    xml_blob = build_sftr_xml(trades)

    # Build a deterministic S3 key, e.g. sftr/2024-11-30.xml
    s3_key = f'sftr/{TODAY}.xml'

    # Upload encrypted (SSE‑KMS) – the same KMS key used for ledger snapshots
    upload_encrypted(xml_blob, s3_key, content_type='application/xml')
    log.info("SFTR report for %s uploaded to s3://%s/%s", TODAY, AUDIT_BUCKET, s3_key)


if __name__ == '__main__':
    main()
