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

# Main decision horizon (must be in HORIZONS_MINUTES)
HORIZON_MINUTES = 60

# How many days back to train
WINDOW_DAYS = 30

# Recency weighting half-life (days)
DECAY_DAYS = 14

# Tax and margin used to define "profitable"
TAX_RATE = 0.02
MARGIN = 0.002  # extra 0.2% above tax for "profitable"

# Features used by all regressors
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

# Minimum samples required to fit a regressor for a given horizon
MIN_SAMPLES_PER_HORIZON = 50

# ----------------------------
# Robustness: label / sigma
# ----------------------------

# Clip targets (future returns) used for training – applied to ALL horizons.
# This removes pathological labels (10x, 20x) while leaving room for real trends.
FUTURE_RETURN_CLIP_LOWER = -0.8   # -80%
FUTURE_RETURN_CLIP_UPPER = +1.0   # +100%

# Sigma calibration slice (for the Normal CDF Win%):
# Only use residuals from a stable regime to estimate sigma_main.
SIGMA_PRICE_MIN_GP = 10_000         # ≥ 10k gp
SIGMA_ABS_RETURN_MAX = 0.50         # |gross future return| ≤ 50%
SIGMA_ABS_NET_HAT_MAX = 0.05        # |predicted net return| ≤ 5%
MIN_SAMPLES_FOR_SIGMA = 1000        # fallback to full if fewer rows


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
      - main-horizon future_return / net_return
      - profit_ok
      - recency-based sample_weight
    """
    df = add_model_features(df_raw)

    # Add multi-horizon returns (targets)
    df = add_multi_horizon_returns(df, horizons_minutes=HORIZONS_MINUTES)

    # Clip all horizon targets to robust bounds (guards against extreme regimes)
    for H in HORIZONS_MINUTES:
        col = f"ret_{H}m"
        if col in df.columns:
            df[col] = df[col].clip(FUTURE_RETURN_CLIP_LOWER, FUTURE_RETURN_CLIP_UPPER)

    # Check we have the main horizon
    col_60 = f"ret_{HORIZON_MINUTES}m"
    if col_60 not in df.columns:
        raise RuntimeError(
            f"Expected column {col_60} after add_multi_horizon_returns()."
        )

    # Require non-missing target at main horizon
    df = df.dropna(subset=[col_60]).copy()

    # Define main 60m future return (already clipped above)
    df["future_return"] = df[col_60]

    # Net return after tax; "profitable" if net_return > MARGIN
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
        reg_models[h].predict(X) ~= ret_{h}m (targets were clipped)
      - Estimate residual std for the main horizon (for probability-of-profit)
        using a stability-trimmed slice.
    """
    X_all = df[FEATURE_COLS].values
    w = df["sample_weight"].values

    reg_models = {}

    # Fit a regressor for each horizon (using clipped targets)
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

    if not reg_models:
        raise RuntimeError("No regression models were trained (not enough data for any horizon).")
    if HORIZON_MINUTES not in reg_models:
        raise RuntimeError(f"Missing regressor for main horizon {HORIZON_MINUTES}m.")

    # ------------------------------
    # Robust sigma_main estimation
    # ------------------------------
    reg_main = reg_models[HORIZON_MINUTES]
    col_60 = f"ret_{HORIZON_MINUTES}m"

    y_true_all = df[col_60].values  # already clipped via add_labels_and_weights
    mask_main = np.isfinite(y_true_all)

    y_true_all = y_true_all[mask_main]
    X_main = X_all[mask_main]
    w_main = w[mask_main]

    y_pred_all = reg_main.predict(X_main)
    mid_all = df.loc[mask_main, "mid_price"].to_numpy()

    # Predicted net return (after tax) for the slice filter
    net_hat_all = (1.0 + y_pred_all) * (1.0 - TAX_RATE) - 1.0

    good_mask = (
        np.isfinite(mid_all) &
        (mid_all >= SIGMA_PRICE_MIN_GP) &
        np.isfinite(y_true_all) &
        (np.abs(y_true_all) <= SIGMA_ABS_RETURN_MAX) &
        np.isfinite(net_hat_all) &
        (np.abs(net_hat_all) <= SIGMA_ABS_NET_HAT_MAX)
    )

    n_good = int(good_mask.sum())
    if n_good < MIN_SAMPLES_FOR_SIGMA:
        print(
            f"[train] Trimmed sigma slice too small ({n_good} rows) – "
            "falling back to full main-horizon sample."
        )
        good_mask = np.isfinite(y_true_all)

    y_true = y_true_all[good_mask]
    y_pred = y_pred_all[good_mask]
    w_h = w_main[good_mask]

    # Normalise weights to avoid numerical issues
    w_h = w_h / max(np.mean(w_h), 1e-8)

    mse = np.average((y_true - y_pred) ** 2, weights=w_h)
    sigma_main = float(np.sqrt(mse))
    sigma_main = max(sigma_main, 1e-8)

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
    key_reg = f"models/remove-noisy-sections/xgb/{date_part}/reg_multi.pkl"
    s3.put_object(Bucket=bucket, Key=key_reg, Body=buf_reg.getvalue())

    # "Latest" pointer (what score_latest.py will load)
    s3.put_object(
        Bucket=bucket,
        Key="models/remove-noisy-sections/xgb/latest_reg.pkl",
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
        # Document robustness settings for transparency
        "label_clip": [FUTURE_RETURN_CLIP_LOWER, FUTURE_RETURN_CLIP_UPPER],
        "sigma_trim": {
            "min_mid_price_gp": SIGMA_PRICE_MIN_GP,
            "max_abs_future_return": SIGMA_ABS_RETURN_MAX,
            "max_abs_net_return_hat": SIGMA_ABS_NET_HAT_MAX,
            "min_rows": MIN_SAMPLES_FOR_SIGMA,
        },
    }

    meta_bytes = json_bytes(meta)
    key_meta = f"models/remove-noisy-sections/xgb/{date_part}/meta.json"
    s3.put_object(Bucket=bucket, Key=key_meta, Body=meta_bytes)
    s3.put_object(
        Bucket=bucket,
        Key="models/remove-noisy-sections/xgb/latest_meta.json",
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
    print(f"Main-horizon residual sigma (trimmed): {sigma_main:.6f}")


if __name__ == "__main__":
    main()
