import os
import io
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
    add_multi_horizon_returns,
    HORIZONS_MINUTES,  # [5, 10, ..., 120]
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

HORIZON_MINUTES = 60      # main decision horizon for classifier / profit metrics
WINDOW_DAYS = 30          # how many days back to train
DECAY_DAYS = 14           # recency weighting half-life (days)
TAX_RATE = 0.02
MARGIN = 0.002            # extra 0.2% above tax for "profitable"
MIN_SAMPLES_PER_HORIZON = 200  # skip horizons with fewer samples than this


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def build_training_dataframe(s3, bucket):
    """
    Pull 5m snapshots for the last WINDOW_DAYS into a single DataFrame.
    """
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=WINDOW_DAYS - 1)).date()
    end_date = now.date()  # include today

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


# ---------------------------------------------------------------------
# Labels & sample weights
# ---------------------------------------------------------------------


def add_labels_and_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features, multi-horizon returns, 60m profit label, and recency weights.
    """
    # Basic per-snapshot features
    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    # Add ret_{H}m columns for all horizons (5..120)
    df = add_multi_horizon_returns(
        df,
        horizons_minutes=HORIZONS_MINUTES,
        tax_rate=TAX_RATE,
    )

    # Ensure we have 60m returns for the classifier
    col_60 = f"ret_{HORIZON_MINUTES}m"
    if col_60 not in df.columns:
        raise RuntimeError(f"Expected column {col_60} after add_multi_horizon_returns().")

    df = df.dropna(subset=[col_60]).copy()
    df["future_return_60m"] = df[col_60]

    # "Profitable" label: require beating tax + margin
    df["profit_ok"] = (df["future_return_60m"] > (TAX_RATE + MARGIN)).astype(int)

    # Recency weights: exponential decay by age in days
    now_ts = df["timestamp"].max()
    age_days = (now_ts - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["sample_weight"] = np.exp(-age_days / DECAY_DAYS)

    return df


# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------


def train_models(df: pd.DataFrame):
    """
    Train:
      - one XGBRegressor per horizon in HORIZONS_MINUTES (multi-h path)
      - one XGBClassifier for 60m "profit_ok"
    """
    # Features used by both regressor(s) and classifier
    feature_cols = ["mid_price", "spread_pct", "log_volume_5m"]

    X = df[feature_cols].values
    w = df["sample_weight"].values

    # Train one regressor per horizon, on rows where that horizon label exists
    reg_models = {}
    for H in HORIZONS_MINUTES:
        col = f"ret_{H}m"
        if col not in df.columns:
            print(f"[train] Horizon {H}m: missing column {col}, skipping.")
            continue

        y_h = df[col].values
        mask = np.isfinite(y_h)

        n_h = mask.sum()
        if n_h < MIN_SAMPLES_PER_HORIZON:
            print(f"[train] Horizon {H}m: only {n_h} samples, skipping.")
            continue

        print(f"[train] Training regressor for {H}m on {n_h} samples.")
        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=4,
        )
        reg.fit(X[mask], y_h[mask], sample_weight=w[mask])
        reg_models[H] = reg

    if not reg_models:
        raise RuntimeError("No regression models were trained (no horizons met sample threshold).")

    # 60m classifier for "profit_ok"
    y_cls = df["profit_ok"].values.astype(int)
    print(f"[train] Training classifier on {len(y_cls)} samples.")
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

    return reg_models, cls, feature_cols


# ---------------------------------------------------------------------
# Save to R2
# ---------------------------------------------------------------------


def save_models(s3, bucket, reg_models, cls, feature_cols):
    """
    Save all horizon regressors (as a dict) + classifier + meta to R2.
    """
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")

    # pickle models into in-memory buffers
    buf_reg = io.BytesIO()
    buf_cls = io.BytesIO()
    joblib.dump(reg_models, buf_reg)
    joblib.dump(cls, buf_cls)
    buf_reg.seek(0)
    buf_cls.seek(0)

    key_reg = f"models/xgb/{date_part}/reg.pkl"
    key_cls = f"models/xgb/{date_part}/cls.pkl"
    s3.put_object(Bucket=bucket, Key=key_reg, Body=buf_reg.getvalue())
    s3.put_object(Bucket=bucket, Key=key_cls, Body=buf_cls.getvalue())

    # latest pointers
    s3.put_object(
        Bucket=bucket,
        Key="models/xgb/latest_reg.pkl",
        Body=buf_reg.getvalue(),
    )
    s3.put_object(
        Bucket=bucket,
        Key="models/xgb/latest_cls.pkl",
        Body=buf_cls.getvalue(),
    )

    # meta with horizon information
    meta = {
        "horizon_minutes": HORIZON_MINUTES,
        "path_horizons_minutes": HORIZONS_MINUTES,
        "trained_horizons": sorted(int(h) for h in reg_models.keys()),
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
    s3.put_object(
        Bucket=bucket,
        Key="models/xgb/latest_meta.json",
        Body=meta_bytes,
    )


def json_bytes(obj):
    import json
    return json.dumps(obj).encode("utf-8")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


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

    reg_models, cls, feature_cols = train_models(df)
    save_models(s3, bucket, reg_models, cls, feature_cols)
    print("Training complete.")


if __name__ == "__main__":
    main()
