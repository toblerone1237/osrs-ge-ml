import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
    add_basic_features,
)

# Config
HORIZON_MINUTES = 60      # how long we plan to hold
WINDOW_DAYS = 30          # how many days back to train
DECAY_DAYS = 14           # recency weighting half-life (days)
TAX_RATE = 0.02
MARGIN = 0.002            # extra 0.2% above tax for "profitable"


def build_training_dataframe(s3, bucket):
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=WINDOW_DAYS)).date()
    end_date = (now - timedelta(days=1)).date()

    chunks = []
    d = start_date
    while d <= end_date:
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        print(d, "keys:", len(keys))
        if keys:
            df_day = flatten_5m_snapshots(s3, bucket, keys)
            if not df_day.empty:
                chunks.append(df_day)
        d += timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    return df


def add_labels_and_weights(df):
    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    df.sort_values(["item_id", "timestamp"], inplace=True)
    buckets_ahead = HORIZON_MINUTES // 5

    df["mid_price_future"] = df.groupby("item_id")["mid_price"].shift(-buckets_ahead)
    df = df.dropna(subset=["mid_price_future"]).copy()

    df["future_return"] = df["mid_price_future"] / df["mid_price"] - 1.0
    df["profit_ok"] = (df["future_return"] > (TAX_RATE + MARGIN)).astype(int)

    # recency weights: exponential decay by age in days
    now = df["timestamp"].max()
    age_days = (now - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["sample_weight"] = np.exp(-age_days / DECAY_DAYS)

    return df


def train_models(df):
    feature_cols = ["mid_price", "spread_pct", "log_volume_5m"]

    X = df[feature_cols].values
    y_reg = df["future_return"].values
    y_cls = df["profit_ok"].values
    w = df["sample_weight"].values

    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=4,
    )
    reg.fit(X, y_reg, sample_weight=w)

    cls = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=4,
    )
    cls.fit(X, y_cls, sample_weight=w)

    return reg, cls, feature_cols


def save_models(s3, bucket, reg, cls, feature_cols):
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")

    # pickle models
    import io
    buf_reg = io.BytesIO()
    buf_cls = io.BytesIO()
    joblib.dump(reg, buf_reg)
    joblib.dump(cls, buf_cls)
    buf_reg.seek(0)
    buf_cls.seek(0)

    key_reg = f"models/xgb/{date_part}/reg.pkl"
    key_cls = f"models/xgb/{date_part}/cls.pkl"
    s3.put_object(Bucket=bucket, Key=key_reg, Body=buf_reg.getvalue())
    s3.put_object(Bucket=bucket, Key=key_cls, Body=buf_cls.getvalue())

    # latest pointers
    s3.put_object(Bucket=bucket, Key="models/xgb/latest_reg.pkl", Body=buf_reg.getvalue())
    s3.put_object(Bucket=bucket, Key="models/xgb/latest_cls.pkl", Body=buf_cls.getvalue())

    meta = {
        "horizon_minutes": HORIZON_MINUTES,
        "window_days": WINDOW_DAYS,
        "decay_days": DECAY_DAYS,
        "feature_cols": feature_cols,
        "tax_rate": TAX_RATE,
        "margin": MARGIN,
        "generated_at_iso": now.isoformat(),
    }
    meta_bytes = json_bytes(meta)
    key_meta = f"models/xgb/{date_part}/meta.json"
    s3.put_object(Bucket=bucket, Key=key_meta, Body=meta_bytes)
    s3.put_object(Bucket=bucket, Key="models/xgb/latest_meta.json", Body=meta_bytes)


def json_bytes(obj):
    import json
    return json.dumps(obj).encode("utf-8")


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    df = build_training_dataframe(s3, bucket)
    if df.empty:
        print("No training data found.")
        return

    df = add_labels_and_weights(df)
    if df.empty:
        print("No rows after labels/weights.")
        return

    reg, cls, feature_cols = train_models(df)
    save_models(s3, bucket, reg, cls, feature_cols)
    print("Training complete.")


if __name__ == "__main__":
    main()
