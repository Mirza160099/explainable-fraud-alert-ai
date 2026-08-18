# Model Evaluation

Fraud detection is an imbalanced-classification problem, so **accuracy alone is not an appropriate primary metric**.

## Metrics used

### ROC-AUC
Useful for overall ranking quality, but can appear optimistic in highly imbalanced problems.

### PR-AUC
Often more informative for fraud because it focuses on performance for the positive class.

### Precision
Of the transactions flagged as fraud, how many were actually fraudulent?

### Recall
Of all fraudulent transactions, how many did the model identify?

### F1
Balances precision and recall.

## Threshold Selection

The example application uses a default 0.50 threshold, but production thresholding should consider:

- investigation capacity
- cost of false positives
- cost of missed fraud
- regulatory / policy requirements
- segment-specific risk
- alert volumes
- analyst SLA

A fraud probability is not the same thing as a final business decision.
