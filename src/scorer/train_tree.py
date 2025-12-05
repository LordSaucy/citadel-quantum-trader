# scripts/train_tree.py
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/training_features.csv')   
X = df.drop(columns=['target'])
y = df['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
bst = xgb.train(params, dtrain, num_boost_round=200,
                evals=[(dval, "validation")],
                early_stopping_rounds=20)

bst.save_model('models/tree_scorer.bin')
# Save the feature order so the runtime knows how to build the vector
import json
json.dump(list(X.columns), open('models/tree_feature_order.json', 'w'))
