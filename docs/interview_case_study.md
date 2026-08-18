# Interview Case Study

## Situation
Fraud teams can receive more alerts than analysts can investigate immediately. A pure risk score is also difficult to trust if analysts cannot understand why an alert was prioritised.

## Task
Build a portfolio workflow that scores synthetic transactions, explains model output, and maps probability into analyst-review priorities.

## Action
- Created a synthetic transaction dataset with behavioural and contextual risk factors.
- Added feature engineering for transaction timing, velocity, account age and device age.
- Trained an XGBoost classifier.
- Evaluated the model using ROC-AUC, PR-AUC, precision, recall and F1.
- Added SHAP explanations for global and individual feature contribution.
- Created analyst-review bands and recommended next actions.
- Built a Streamlit interface for interactive alert review.
- Documented why model output must remain decision support rather than an autonomous fraud judgement.

## Result
Produced an explainable fraud-alert prioritisation portfolio project demonstrating machine learning, security/risk analytics, model explainability and analyst workflow design.

## Interview Talking Points
1. Why accuracy is weak for fraud detection.
2. Precision vs recall trade-off.
3. Why PR-AUC is useful.
4. How to choose an alert threshold.
5. What SHAP explains—and what it does not.
6. Why analysts should remain in the loop.
7. Data leakage.
8. Model drift.
9. Class imbalance.
10. How to monitor a deployed fraud model.
11. How to incorporate analyst feedback.
12. Ethical and regulatory considerations.
