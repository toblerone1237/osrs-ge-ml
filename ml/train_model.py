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
    add_model_features,
    add_multi_horizon_returns,
    HORIZONS_MINUTES,  # [5, 10, ..., 120]
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HORIZON_MINUTES = 60      # main decision horizon (must be in HORIZONS_MINUTES)
WINDOW_DAYS = 30          # how many days back to train
DECAY_DAYS = 14           # recency weighting half-life (days)
TAX_RATE = 0.02
MARGIN = 0.002            # extra 0.2% above tax for "profitable"

# Features used by both regressors and probability-of-profit
FEATURE_COLS = [
    "mid_price",
    "spread_pct",
    "log_volume_5m",
    "total_volume_5m",
    "ret_5m_past",
    "ret_15m_past",
    "ret_60m_past",
    "volatility_60m",
    "log_rolling_volume_60m",
    "volume_ratio_5m_to_60m",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# how many samples we require to train a regressor for a given horizon
MIN_SAMPLES_PER_HORIZON = 50


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_training_dataframe(s3, bucket):
    """
    Load raw 5-minute snapshots from the last WINDOW_DAYS into a single DataFrame.
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


# ---------------------------------------------------------------------------
# Labels & weights
# ---------------------------------------------------------------------------

def add_labels_and_weights(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - model features
      - multi-horizon return columns: ret_5m, ret_10m, ..., ret_120m
      - 60m "profit_ok" label
      - recency-based sample_weight
    """
    df = add_model_features(df_raw)

    # Add multi-horizon returns (targets)
    df = add_multi_horizon_returns(df, horizons_minutes=HORIZONS_MINUTES)

    # Check we have the main horizon
    col_60 = f"ret_{HORIZON_MINUTES}m"
    if col_60 not in df.columns:
        raise RuntimeError(
            f"Expected column {col_60} after add_multi_horizon_returns()."
        )

    # Require non-missing target at main horizon
    df = df.dropna(subset=[col_60]).copy()

    # Define main 60m future return
    df["future_return"] = df[col_60]

    # "Profitable" event: net return after tax + margin > 0
    # net_return = (1 + future_return) * (1 - TAX_RATE) - 1
    df["net_return"] = (1.0 + df["future_return"]) * (1.0 - TAX_RATE) - 1.0
    df["profit_ok"] = (df["net_return"] > MARGIN).astype(int)

    # Recency weights: exponential decay by age in days
    now_ts = df["timestamp"].max()
    age_days = (now_ts - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["sample_weight"] = np.exp(-age_days / DECAY_DAYS)

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame):
    """
    Train:
      - A dict of XGBRegressors, one per horizon in HORIZONS_MINUTES
        reg_models[h].predict(X) ~= ret_{h}m
      - Estimate residual std for the main horizon (for probability-of-profit)
    """
    X_all = df[FEATURE_COLS].values
    w = df["sample_weight"].values

    reg_models = {}
    sigma_main = None

    for H in HORIZONS_MINUTES:
        col = f"ret_{H}m"
        if col not in df.columns:
            print(f"[train] Horizon {H}m: missing {col}, skipping.")
            continue

        y_h = df[col].values
        mask = np.isfinite(y_h)
        n = int(mask.sum())
        if n < MIN_SAMPLES_PER_HORIZON:
            print(f"[train] Horizon {H}m: only {n} samples, skipping.")
            continue

        print(f"[train] Training regressor for {H}m on {n} samples.")
        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=4,
            tree_method="hist",
        )
        reg.fit(X_all[mask], y_h[mask], sample_weight=w[mask])
        reg_models[H] = reg

        # After fitting the main horizon model, compute residual std
        if H == HORIZON_MINUTES:
            y_true = y_h[mask]
            y_pred = reg.predict(X_all[mask])
            w_h = w[mask]
            # Normalise weights to avoid numerical issues
            w_h = w_h / np.mean(w_h)
            mse = np.average((y_true - y_pred) ** 2, weights=w_h)
            sigma = float(np.sqrt(mse))
            sigma_main = max(sigma, 1e-8)  # avoid degenerate zero

    if not reg_models:
        raise RuntimeError("No regression models were trained (not enough data for any horizon).")

    if sigma_main is None:
        raise RuntimeError(f"No residual std computed for main horizon {HORIZON_MINUTES}m.")

    return reg_models, sigma_main, FEATURE_COLS


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_models(s3, bucket, reg_models, sigma_main, feature_cols):
    """
    Save:
      - reg_models (dict: horizon -> regressor) as latest_reg.pkl
      - meta.json with horizon, feature, and distribution info
    """
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")

    # Serialize models into memory buffer
    buf_reg = io.BytesIO()
    joblib.dump(reg_models, buf_reg)
    buf_reg.seek(0)

    # Date-stamped key
    key_reg = f"models/xgb/{date_part}/reg_multi.pkl"
    s3.put_object(Bucket=bucket, Key=key_reg, Body=buf_reg.getvalue())

    # "Latest" pointer (what score_latest.py will load)
    s3.put_object(
        Bucket=bucket,
        Key="models/xgb/latest_reg.pkl",
        Body=buf_reg.getvalue(),
    )

    # Metadata
    meta = {
        "horizon_minutes": HORIZON_MINUTES,          # main horizon (60m)
        "path_horizons_minutes": HORIZONS_MINUTES,   # [5,10,...,120]
        "window_days": WINDOW_DAYS,
        "decay_days": DECAY_DAYS,
        "feature_cols": feature_cols,                # used for regressors
        "tax_rate": TAX_RATE,
        "margin": MARGIN,
        "sigma_main": float(sigma_main),
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    df_raw = build_training_dataframe(s3, bucket)
    if df_raw.empty:
        print("No training data found.")
        return

    df = add_labels_and_weights(df_raw)
    if df.empty:
        print("No rows after labels/weights.")
        return

    reg_models, sigma_main, feature_cols = train_models(df)
    save_models(s3, bucket, reg_models, sigma_main, feature_cols)
    print("Training complete.")
    print(f"Main-horizon residual sigma: {sigma_main:.6f}")


if __name__ == "__main__":
    main()
