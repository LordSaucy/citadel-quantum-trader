import redis
import pandas as pd
import os

r = redis.StrictRedis(host=os.getenv('REDIS_HOST'),
                      port=int(os.getenv('REDIS_PORT')),
                      db=int(os.getenv('REDIS_DB')),
                      decode_responses=True)

def enrich_with_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row (timestamp) we look up the most recent sentiment
    value from Redis and attach it as a new feature.
    """
    # Assume df.index is a pandas DatetimeIndex (UTC)
    sentiment_vals = []
    for ts in df.index:
        # Find the latest sentiment key before `ts`
        # Redis does not support range queries on keys, so we store the
        # *latest* sentiment under a fixed key that we constantly overwrite.
        # Simpler: keep a single key `sentiment:latest` that the ingestor
        # updates after each fetch.
        val = r.get('sentiment:latest')
        sentiment_vals.append(float(val) if val else 0.0)   # default neutral
    df['sentiment_score'] = sentiment_vals
    return df
