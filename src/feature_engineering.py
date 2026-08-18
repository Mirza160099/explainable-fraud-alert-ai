import pandas as pd

FEATURES = [
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
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_log1p"] = (out["amount"] + 1).apply(__import__("math").log)
    out["night_transaction"] = out["hour"].isin([0,1,2,3,4,5]).astype(int)
    out["high_velocity"] = (out["transactions_1h"] >= 6).astype(int)
    out["very_new_account"] = (out["account_age_days"] < 30).astype(int)
    out["very_new_device"] = (out["device_age_days"] < 7).astype(int)
    return out
