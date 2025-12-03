import os
import json
import redis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------------------------------------------------
# Initialise Redis client (fast key‑value store)
# -----------------------------------------------------------------
r = redis.StrictRedis(host=os.getenv('REDIS_HOST'),
                      port=int(os.getenv('REDIS_PORT')),
                      db=int(os.getenv('REDIS_DB')),
                      decode_responses=True)

# -----------------------------------------------------------------
# Choose model (VADER = lightweight, FinBERT = more accurate)
# -----------------------------------------------------------------
MODEL = os.getenv('SENTIMENT_MODEL', 'vader').lower()

if MODEL == 'finbert':
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    def get_score(text: str) -> float:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # FinBERT outputs [negative, neutral, positive]
        return float(probs[2] - probs[0])   # map to -1 .. +1
else:   # VADER fallback
    analyzer = SentimentIntensityAnalyzer()
    def get_score(text: str) -> float:
        # VADER returns a compound score in [-1, 1]
        return analyzer.polarity_scores(text)['compound']
