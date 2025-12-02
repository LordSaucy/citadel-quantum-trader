import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import json

df = pd.read_csv('data/regime_training.csv')
X = df[['ema20','ema50','atr_1h','close']]
y = df['regime']

clf = LogisticRegression(multi_class='multinomial', max_iter=500)
clf.fit(X, y)

# Persist the model (pickle) and the feature order
joblib.dump(clf, 'models/regime_classifier.pkl')
json.dump(list(X.columns), open('models/regime_features.json','w'))
