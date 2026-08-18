from pathlib import Path
import json
import joblib
import pandas as pd
import shap

MODEL_DIR = Path("models")
model = joblib.load(MODEL_DIR / "fraud_xgb.joblib")
features = json.loads((MODEL_DIR / "features.json").read_text())

df = pd.read_csv("data/processed/model_scored_test_set.csv")
X = df[features].head(300)

explainer = shap.TreeExplainer(model)
values = explainer.shap_values(X)

# Save a compact feature-importance table rather than assuming image rendering
importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": abs(values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

importance.to_csv("data/processed/shap_feature_importance.csv", index=False)
print(importance.head(10))
