# Explainable Fraud Alert AI

An explainable machine-learning portfolio project for **fraud-alert prioritisation and analyst decision support** using Python, XGBoost, SHAP and Streamlit.

> **Important:** This project uses synthetic data and is designed for learning/portfolio purposes. It is not a production fraud-detection system and must not be represented as autonomous financial-crime decisioning.

## Business Problem

Fraud teams often face two challenges:

1. **Too many alerts** for analysts to review immediately.
2. **Low interpretability** when a model only outputs a risk score.

This project addresses both by:

- predicting fraud probability
- prioritising alerts into review bands
- explaining feature contribution with SHAP
- exposing results through an analyst-facing Streamlit workflow

## Tech Stack

- Python
- Pandas
- NumPy
- XGBoost
- scikit-learn
- SHAP
- Streamlit
- Joblib

## Repository Structure

```text
explainable-fraud-alert-ai/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── explain.py
│   └── analyst_review.py
├── app/
│   └── app.py
├── models/
├── docs/
├── notebooks/
├── screenshots/
├── requirements.txt
└── README.md
```

## Features

The synthetic dataset includes:

- transaction amount
- transaction hour
- country risk
- merchant risk
- device age
- account age
- failed logins
- short-term transaction velocity
- 24-hour transaction velocity
- new-device flag
- cross-border flag

Feature engineering adds:

- log-transformed amount
- night-transaction flag
- high-velocity flag
- very-new-account flag
- very-new-device flag

## Model

The project uses **XGBoost** because tree-based boosting works well for structured/tabular risk data and is compatible with SHAP TreeExplainer.

## Evaluation

The training workflow reports:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1

Fraud detection is typically imbalanced, so the project intentionally avoids using raw accuracy as the main success metric.

See [`docs/model_evaluation.md`](docs/model_evaluation.md).

## Explainability

SHAP is used to explain which features most influenced a risk score.

Example analyst questions:

- Was the transaction amount unusually high?
- Did a new device increase risk?
- Was transaction velocity abnormal?
- Did country/merchant risk increase the score?
- Were repeated authentication failures relevant?

See [`docs/shap_explainability.md`](docs/shap_explainability.md).

## Analyst Review Bands

Example portfolio logic:

```text
P1: probability >= 0.80
P2: probability >= 0.60
P3: probability >= 0.35
P4: probability < 0.35
```

These are demonstration thresholds only.

A production fraud operation should choose thresholds based on:

- investigation capacity
- business-loss tolerance
- false-positive cost
- missed-fraud cost
- regulatory requirements
- customer impact
- risk segment

## Streamlit Workflow

The app lets an analyst:

1. Enter transaction/context features.
2. Generate a fraud probability.
3. View a review-priority band.
4. View a suggested analyst action.
5. Inspect the strongest SHAP feature contributions.

Run:

```bash
pip install -r requirements.txt
python src/train_model.py
python src/explain.py
streamlit run app/app.py
```

## Analyst-in-the-Loop Design

The project is deliberately framed as **decision support**, not automated guilt/fraud determination.

Model output should be combined with:

- analyst review
- transaction/customer context
- policy rules
- additional data sources
- regulatory processes
- human escalation

## Production Extensions

A production-quality system would need:

- feature store / governed feature pipeline
- model registry
- drift monitoring
- bias/fairness assessment
- threshold monitoring
- analyst feedback loop
- alert/case management integration
- role-based access control
- audit logging
- secure API serving
- versioned deployment
- privacy and retention controls

## Skills Demonstrated

- Fraud analytics
- Risk analytics
- Machine learning
- XGBoost
- SHAP
- Explainable AI
- Python
- Feature engineering
- Model evaluation
- Streamlit
- Analyst workflow design
- Security / financial-crime decision support

## Interview Talking Points

1. Why class imbalance matters.
2. Precision vs recall.
3. ROC-AUC vs PR-AUC.
4. Threshold tuning.
5. SHAP interpretation.
6. Model drift.
7. Data leakage.
8. Human-in-the-loop decisions.
9. False-positive management.
10. Production model monitoring.
11. Analyst feedback.
12. Explainability limitations.

## Portfolio Classification

**Type:** Completed/Expanded Portfolio Project  
**Data:** Synthetic  
**Purpose:** Demonstrate explainable fraud-risk analytics and analyst decision support.
