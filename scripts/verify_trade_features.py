# scripts/verify_trade_features.py
import json, psycopg2, pandas as pd

conn = psycopg2.connect(dsn=os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute("SELECT id, feature_blob FROM trades ORDER BY id DESC LIMIT 10")
rows = cur.fetchall()

for trade_id, blob in rows:
    feats = json.loads(blob)
    # Example sanity check: ATR stop should never be > entry_price
    if feats['atr_stop'] > feats['entry_price']:
        print(f"⚠️ Trade {trade_id}: atr_stop > entry_price")
