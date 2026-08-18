# SHAP Explainability

SHAP provides feature-attribution values that help explain how individual features influenced a model prediction.

## Why this matters in fraud analytics

Analysts often need more than a risk score. They need context such as:

- unusually high transaction amount
- new device
- cross-border activity
- elevated country or merchant risk
- repeated failed logins
- high transaction velocity

## Limitations

SHAP explains the model, not objective truth.

A high SHAP contribution does not prove causation, criminal intent, or fraud.

Production use requires:
- feature governance
- bias/fairness review
- model monitoring
- drift monitoring
- human review
- documentation
- validation
