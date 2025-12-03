#!/usr/bin/env python3
"""
Form PF – quarterly reporting for US investment advisers.
Creates a CSV file that matches the SEC’s Form PF layout (simplified version).
"""

import os, datetime, csv, io, logging
import psycopg2
from report_utils import upload_encrypted

log = logging.getLogger('cqt.formpf')
DB_DSN = os.getenv('POSTGRES_DSN')
AUDIT_BUCKET = os.getenv('AUDIT_S3_BUCKET', 'citadel-audit')
KMS_ENCRYPT_KEY = os.getenv('AWS_KMS_S3_KEY')

def fetch_quarterly_stats():
    """Aggregate performance & risk metrics for the past quarter."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    # Determine quarter boundaries
    today = datetime.date.today()
    quarter = (today.month - 1) // 3 + 1
    start_month = 3 * (quarter - 1) + 1
    start_date = datetime.date(today.year, start_month, 1)
    end_date = today

    cur.execute("""
        SELECT
            SUM(equity) AS total_aum,
            AVG(pnl)   AS avg_daily_pnl,
            STDDEV(pnl) AS pnl_stddev,
            COUNT(*)   AS trade_count,
            MAX(drawdown_pct) AS max_drawdown
        FROM equity_history
        WHERE date BETWEEN %s AND %s
    """, (start_date, end_date))

    row = cur.fetchone()
    cur.close()
    conn.close()
    return {
        'quarter'        : f"{today.year}Q{quarter}",
        'total_aum'      : float(row[0] or 0),
        'avg_daily_pnl'  : float(row[1] or 0),
        'pnl_stddev'     : float(row[2] or 0),
        'trade_count'    : int(row[3] or 0),
        'max_drawdown'   : float(row[4] or 0),
    }

def build_csv(stats):
    """Return a CSV byte string matching the simplified Form PF columns."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header – you can expand this to the full SEC column list
    writer.writerow([
        'Quarter',
        'Total_AUM_USD',
        'Avg_Daily_PnL_USD',
        'PnL_StdDev_USD',
        'Trade_Count',
        'Max_Drawdown_Pct'
    ])

    writer.writerow([
        stats['quarter'],
        f"{stats['total_aum']:.2f}",
        f"{stats['avg_daily_pnl']:.2f}",
        f"{stats['pnl_stddev']:.2f}",
        stats['trade_count'],
        f"{stats['max_drawdown']:.4f}"
    ])

    return output.getvalue().encode('utf-8')

def main():
    stats = fetch_quarterly_stats()
    csv_blob = build_csv(stats)

    s3_key = f'form_pf/{stats["quarter"]}.csv'
    upload_encrypted(csv_blob, s3_key, content_type='text/csv')
    log.info("Form PF %s uploaded to s3://%s/%s", stats['quarter'], AUDIT_BUCKET, s3_key)


if __name__ == '__main__':
    main()
