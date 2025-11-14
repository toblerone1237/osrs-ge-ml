import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import io

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
    add_basic_features,
)


def get_latest_5m_key(s3, bucket):
    now = datetime.now(timezone.utc)
    # try today, then yesterday
    for delta in [0, 1]:
        d = (now - pd.Timedelta(days=delta)).date()
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if keys:
            keys.sort()
            return keys[-1]
    return None


def load_models_and_meta(s3, bucket):
    reg_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_reg.pkl")
    cls_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_cls.pkl")
    meta_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_meta.json")

    # Read raw bytes from S3
    reg_bytes = reg_obj["Body"].read()
    cls_bytes = cls_obj["Body"].read()

    # Wrap in BytesIO so joblib can seek
    reg = joblib.load(io.BytesIO(reg_bytes))
    cls = joblib.load(io.BytesIO(cls_bytes))

    meta = json.loads(meta_obj["Body"].read())
    return reg, cls, meta


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    latest_key = get_latest_5m_key(s3, bucket)
    if not latest_key:
        print("No 5m snapshot found.")
        return

    print("Using snapshot:", latest_key)
    df = flatten_5m_snapshots(s3, bucket, [latest_key])
    if df.empty:
        print("Snapshot empty.")
        return

    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0]

    reg, cls, meta = load_models_and_meta(s3, bucket)
    feature_cols = meta["feature_cols"]
    H = meta["horizon_minutes"]
    tax = meta["tax_rate"]

    X = df[feature_cols].values

    future_ret = reg.predict(X)
    prob_profit = cls.predict_proba(X)[:, 1]

    df["future_return_hat"] = future_ret
    df["prob_profit"] = prob_profit

    mid = df["mid_price"].values
    mid_future = mid * (1 + future_ret)
    buy_price = mid
    net_sell = mid_future * (1 - tax)
    profit = net_sell - buy_price
    hold_seconds = H * 60
    profit_per_sec = profit / hold_seconds

    df["expected_profit"] = profit
    df["expected_profit_per_second"] = profit_per_sec

    mask = (df["expected_profit"] > 0) & (df["prob_profit"] > 0.55)
    candidates = df[mask].copy()
    candidates.sort_values("expected_profit_per_second", ascending=False, inplace=True)

    signals = []
    for _, row in candidates.head(200).iterrows():
        signals.append(
            {
                "item_id": int(row["item_id"]),
                "mid_now": float(row["mid_price"]),
                "future_return_hat": float(row["future_return_hat"]),
                "prob_profit": float(row["prob_profit"]),
                "expected_profit": float(row["expected_profit"]),
                "expected_profit_per_second": float(row["expected_profit_per_second"]),
                "hold_minutes": H,
            }
        )

    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")
    time_part = now.strftime("%H-%M")

    out = {
        "generated_at_iso": now.isoformat(),
        "horizon_minutes": H,
        "tax_rate": tax,
        "signals": signals,
    }
    body = json.dumps(out).encode("utf-8")

    latest_key = "signals/latest.json"
    dated_key = f"signals/{date_part}/{time_part}.json"

    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")

    print(f"Wrote {len(signals)} signals.")


if __name__ == "__main__":
    main()
