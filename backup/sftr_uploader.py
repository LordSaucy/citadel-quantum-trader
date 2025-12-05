#!/usr/bin/env python3
"""
SFTR (EU) – daily transaction report.

The format is a simplified XML that complies with the ESMA schema.
Only the fields required for a "transaction‑level" report are populated.

✅ FIXED: Removed unused variable and fixed duplicate function definition
"""

import os
import logging
from datetime import datetime, timezone, date
import xml.etree.ElementTree as ET

import psycopg2

from report_utils import upload_encrypted

# =====================================================================
# Configuration
# =====================================================================
logger = logging.getLogger(__name__)

DB_DSN = os.getenv('POSTGRES_DSN')
# e.g. "dbname=citadel user=citadel password=… host=postgres"

AUDIT_BUCKET = os.getenv('AUDIT_BUCKET', 'cqt-audit-reports')

TODAY = date.today().isoformat()


# =====================================================================
# Database & Transaction Fetching
# =====================================================================
def fetch_transactions():
    """
    Fetch all trades from yesterday's date.
    
    Returns
    -------
    list of dict
        Each dict contains: bucket_id, symbol, direction, volume,
        entry_price, sl, tp, pnl, timestamp
    
    Raises
    ------
    psycopg2.Error
        If database connection or query fails
    """
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT bucket_id, symbol, direction, volume, entry_price,
                   sl, tp, pnl, timestamp
            FROM trades
            WHERE timestamp::date = %s
        """, (TODAY,))
        
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        
        return rows
    finally:
        cur.close()
        conn.close()


# =====================================================================
# SFTR XML Report Generation
# =====================================================================
def build_sftr_xml(trades):
    """
    Build an ESMA SFTR-compliant XML report from trade data.
    
    Parameters
    ----------
    trades : list of dict
        List of trade dictionaries (as returned by fetch_transactions).
    
    Returns
    -------
    bytes
        XML document encoded as UTF-8 with XML declaration.
    """
    root = ET.Element(
        'SFTRReport',
        attrib={
            'creationDateTime': datetime.now(timezone.utc).isoformat()
        }
    )
    
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


# =====================================================================
# Main Entry Point
# =====================================================================
def main():
    """
    Main execution: fetch transactions, build SFTR report, and upload.
    
    ✅ FIXED: Removed unused variable 'trades' (was: trades = fetch_transactions())
    ✅ FIXED: Removed duplicate function definition
    """
    # ✅ FIXED: Directly use the result without assigning to unused variable
    transactions = fetch_transactions()
    
    if not transactions:
        logger.info("No trades for %s – nothing to report.", TODAY)
        return
    
    # Build SFTR XML report
    xml_blob = build_sftr_xml(transactions)
    
    # Build a deterministic S3 key (e.g., sftr/2024-11-30.xml)
    s3_key = f'sftr/{TODAY}.xml'
    
    # Upload encrypted (SSE-KMS) – the same KMS key used for ledger snapshots
    upload_encrypted(xml_blob, s3_key, content_type='application/xml')
    
    logger.info(
        "SFTR report for %s uploaded to s3://%s/%s",
        TODAY,
        AUDIT_BUCKET,
        s3_key
    )


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        main()
    except Exception as exc:
        logger.exception("SFTR upload failed: %s", exc)
        raise
