df_raw = market_feed.get_recent_dataframe()   # OHLC + volume
df = enrich_with_sentiment(df_raw)

# Existing LSTM/HMM expects columns: ['open','high','low','close','volume','sentiment_score']
regime = regime_model.predict(df)   # returns 'bull', 'neutral', 'bear'
