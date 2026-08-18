from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from feature_engineering import build_features

DATA = Path("data/raw/fraud_transactions_synthetic.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df = build_features(df)

features = [
    "amount",
    "hour",
    "country_risk_score",
    "device_age_days",
    "account_age_days",
    "failed_logins_24h",
    "transactions_1h",
    "velocity_24h",
    "is_new_device",
    "is_cross_border",
    "merchant_risk_score",
    "amount_log1p",
    "night_transaction",
    "high_velocity",
    "very_new_account",
    "very_new_device",
]

X = df[features]
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=220,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.50).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, pred, average="binary", zero_division=0
)

metrics = {
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    "precision_at_0_50": round(float(precision), 4),
    "recall_at_0_50": round(float(recall), 4),
    "f1_at_0_50": round(float(f1), 4),
    "test_rows": int(len(y_test)),
    "fraud_rate_test": round(float(y_test.mean()), 4),
}

joblib.dump(model, MODEL_DIR / "fraud_xgb.joblib")
(MODEL_DIR / "features.json").write_text(json.dumps(features, indent=2))
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

results = X_test.copy()
results["fraud_label"] = y_test.values
results["fraud_probability"] = proba
results.to_csv("data/processed/model_scored_test_set.csv", index=False)

print(metrics)
