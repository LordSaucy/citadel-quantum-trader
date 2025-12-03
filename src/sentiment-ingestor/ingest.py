import feedparser, requests, time, datetime, yaml, schedule, logging
from sentiment import get_score, r

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# -----------------------------------------------------------------
# 1️⃣  RSS news ingestion
# -----------------------------------------------------------------
def fetch_rss():
    for url in cfg['rss_feeds']:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:   # only newest 5 items
                # Use the GUID or link as a unique key
                uid = entry.get('id') or entry.link
                # Combine title + summary for sentiment
                text = f"{entry.title}. {entry.summary}"
                score = get_score(text)
                # Store in Redis with a TTL (e.g., 24 h)
                r.setex(f"sentiment:news:{uid}", cfg['redis']['ttl_seconds'], score)
                logging.info(f"RSS {url} → {uid[:8]} score={score:.3f}")
        except Exception as e:
            logging.error(f"RSS fetch error ({url}): {e}")

# -----------------------------------------------------------------
# 2️⃣  Macro calendar ingestion (FRED CPI example)
# -----------------------------------------------------------------
def fetch_macro():
    api_key = cfg['macro']['fred_api_key']
    series = cfg['macro']['series_id']
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series}&api_key={api_key}&observation_start=1900-01-01"
           "&frequency=m&output_type=1&file_type=json")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Grab the most recent observation
        latest = data['observations'][-1]
        value = float(latest['value'])
        # Simple transformation: convert CPI % change to sentiment
        # (higher CPI → bearish for equities, so we map to negative)
        # Normalize around a 2‑% baseline:
        baseline = 2.0
        delta = (value - baseline) / baseline   # e.g., 2.5% → +0.25
        score = -max(min(delta, 1.0), -1.0)      # clamp to [-1, 1], invert sign
        r.setex(f"sentiment:macro:{series}", cfg['redis']['ttl_seconds'], score)
        logging.info(f"CPI {value:.2f}% → macro score={score:.3f}")
    except Exception as e:
        logging.error(f"Macro fetch error: {e}")

# -----------------------------------------------------------------
# 3️⃣  Scheduler – run every minute for RSS, every 30 min for macro
# -----------------------------------------------------------------
schedule.every(1).minutes.do(fetch_rss)
schedule.every(cfg['macro']['poll_interval_minutes']).minutes.do(fetch_macro)

logging.info("Sentiment ingestor started – entering schedule loop")
while True:
    schedule.run_pending()
    time.sleep(1)
