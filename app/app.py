from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import shap
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from feature_engineering import build_features
from analyst_review import review_band, recommended_action

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Explainable Fraud Alert AI", layout="wide")
st.title("Explainable Fraud Alert Prioritisation")
st.caption("Portfolio demo using synthetic data. Not a production fraud decisioning system.")

if not (MODEL_DIR / "fraud_xgb.joblib").exists():
    st.warning("Train the model first with: python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_DIR / "fraud_xgb.joblib")
features = json.loads((MODEL_DIR / "features.json").read_text())

with st.sidebar:
    st.header("Transaction Inputs")
    amount = st.number_input("Amount", min_value=1.0, value=250.0)
    hour = st.slider("Hour", 0, 23, 2)
    country_risk_score = st.slider("Country risk score", 0.0, 1.0, 0.35)
    device_age_days = st.number_input("Device age (days)", min_value=0, value=3)
    account_age_days = st.number_input("Account age (days)", min_value=1, value=120)
    failed_logins_24h = st.number_input("Failed logins (24h)", min_value=0, value=2)
    transactions_1h = st.number_input("Transactions (1h)", min_value=0, value=5)
    velocity_24h = st.number_input("Transactions (24h)", min_value=0, value=14)
    is_new_device = st.selectbox("New device?", [0,1], format_func=lambda x: "Yes" if x else "No")
    is_cross_border = st.selectbox("Cross-border?", [0,1], format_func=lambda x: "Yes" if x else "No")
    merchant_risk_score = st.slider("Merchant risk score", 0.0, 1.0, 0.4)

row = pd.DataFrame([{
    "amount": amount,
    "hour": hour,
    "country_risk_score": country_risk_score,
    "device_age_days": device_age_days,
    "account_age_days": account_age_days,
    "failed_logins_24h": failed_logins_24h,
    "transactions_1h": transactions_1h,
    "velocity_24h": velocity_24h,
    "is_new_device": is_new_device,
    "is_cross_border": is_cross_border,
    "merchant_risk_score": merchant_risk_score,
}])

row = build_features(row)
prob = float(model.predict_proba(row[features])[:,1][0])

c1, c2, c3 = st.columns(3)
c1.metric("Fraud Probability", f"{prob:.1%}")
c2.metric("Review Band", review_band(prob))
c3.metric("Suggested Action", recommended_action(prob, is_new_device, failed_logins_24h))

st.subheader("Why was this alert scored this way?")
explainer = shap.TreeExplainer(model)
vals = explainer.shap_values(row[features])[0]

explain_df = pd.DataFrame({
    "feature": features,
    "feature_value": row[features].iloc[0].values,
    "shap_contribution": vals,
})
explain_df["abs_contribution"] = explain_df["shap_contribution"].abs()
explain_df = explain_df.sort_values("abs_contribution", ascending=False)

st.dataframe(explain_df[["feature","feature_value","shap_contribution"]].head(10), use_container_width=True)

st.info(
    "This model is for portfolio demonstration only. In a real financial-crime environment, "
    "model outputs should support—not replace—analyst judgement, policy controls, regulatory requirements, "
    "and human review."
)
