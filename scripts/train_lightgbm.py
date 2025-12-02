#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# -----------------------------------------------------------------
# 1️⃣ Load the training data – this should be the same CSV/Parquet
#    you already use for XGBoost (features + label)
# -----------------------------------------------------------------
DATA_PATH = "data/training_features.parquet"   # adjust as needed
df = pd.read_parquet(DATA_PATH)

X = df.drop(columns=["target"]).astype(np.float32)   # all engineered features
y = df["target"].astype(int)                         # binary win/loss (or multi‑class)

# -----------------------------------------------------------------
# 2️⃣ Train / validation split
# -----------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 3️⃣ LightGBM parameters – start simple, then tune
# -----------------------------------------------------------------
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

bst = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    early_stopping_rounds=30,
    verbose_eval=False,
)

# -----------------------------------------------------------------
# 4️⃣ Evaluate (optional but useful)
# -----------------------------------------------------------------
val_pred = bst.predict(X_val)
auc = roc_auc_score(y_val, val_pred)
acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
print(f"Validation AUC: {auc:.4f}  Accuracy: {acc:.4f}")

# -----------------------------------------------------------------
# 5️⃣ Persist the model – LightGBM’s native binary format is fastest
# -----------------------------------------------------------------
MODEL_PATH = "models/lightgbm_model.txt"
bst.save_model(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Also dump a pickle for quick loading in Python (optional)
joblib.dump(bst, "models/lightgbm_model.pkl")
