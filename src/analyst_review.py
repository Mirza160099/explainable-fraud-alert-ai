def review_band(probability: float) -> str:
    if probability >= 0.80:
        return "P1 - Immediate analyst review"
    if probability >= 0.60:
        return "P2 - High-priority review"
    if probability >= 0.35:
        return "P3 - Standard review"
    return "P4 - Monitor / lower priority"

def recommended_action(probability: float, is_new_device: int, failed_logins_24h: int) -> str:
    if probability >= 0.80 and (is_new_device or failed_logins_24h >= 3):
        return "Escalate for account/transaction investigation"
    if probability >= 0.60:
        return "Analyst validation with additional context"
    if probability >= 0.35:
        return "Queue for standard review"
    return "No immediate escalation"
