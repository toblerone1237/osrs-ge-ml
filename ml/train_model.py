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

EXPERIMENT = "quantile"

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

# Main decision horizon in minutes (used for sigma_main)
HORIZON_MINUTES = 60

# How many days of 5m snapshots to train on
WINDOW_DAYS = 30

# Recency decay (days) for sample weights
DECAY_DAYS = 14

# Economic parameters (must match score_latest.py and analyse_sigma_buckets.py)
TAX_RATE = 0.02
MARGIN = 0.002

# Feature columns for the regressors (built by add_model_features)
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

# Minimum samples per horizon per regime
MIN_SAMPLES_PER_HORIZON = 50

# ---------------------------------------------------------------------------
# Regime definitions and label clipping
# ---------------------------------------------------------------------------

# Regimes by mid_price (gp)
# These are also saved into meta and reused at score/eval time.
REGIME_DEFS = {
    "low": {
        "mid_price_min": 0,
        "mid_price_max": 10_000,
    },
    "mid": {
        "mid_price_min": 10_000,
        "mid_price_max": 100_000,
    },
    "high": {
        "mid_price_min": 100_000,
        "mid_price_max": None,  # no upper bound
    },
}

# Per‑regime clipping for training targets (returns are in decimal form)
REGIME_CLIPPING = {
    "high": {"min": -0.40, "max": 0.40},   # ±40%
    "mid":  {"min": -0.60, "max": 0.60},   # ±60%
    "low":  {"min": -1.50, "max": 1.50},   # ±150%
}


def assign_regime_for_price(mid_price: float, regime_defs=None) -> str:
    """
    Map a mid_price to a regime name using regime_defs (defaults to REGIME_DEFS).

    Falls back to "mid" if price is missing or doesn't match any range.
    """
    if regime_defs is None:
        regime_defs = REGIME_DEFS

    try:
        p = float(mid_price)
    except (TypeError, ValueError):
        return "mid"

    if not np.isfinite(p) or p <= 0:
        return "mid"

    for name, bounds in regime_defs.items():
        lo = bounds.get("mid_price_min", 0.0)
        hi = bounds.get("mid_price_max", None)
        if hi is None:
            if p >= lo:
                return name
        else:
            if lo <= p < hi:
                return name

    return "mid"


def clip_returns_for_regime(y: np.ndarray, regime_name: str) -> np.ndarray:
    """
    Apply regime-specific clipping to a vector of returns.

    y is expected to be a numpy array of decimal returns (e.g. 0.05 = +5%).
    """
    cfg = REGIME_CLIPPING.get(regime_name)
    if cfg is None:
        return y

    ymin = cfg["min"]
    ymax = cfg["max"]
    return np.clip(y, ymin, ymax)


# ---------------------------------------------------------------------------
# Data loading and labelling
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
# Training loop (multi-regime)
# ---------------------------------------------------------------------------


def train_models(df: pd.DataFrame):
    """
    Train multi-regime, multi-horizon XGB regressors.

    Returns:
      - reg_models: dict[regime_name][horizon_minutes] -> XGBRegressor
      - sigma_main_per_regime: dict[regime_name] -> float residual std at main horizon
      - feature_cols: the list of feature column names used
    """
    df = df.copy()
    df["regime"] = df["mid_price"].apply(assign_regime_for_price)

    reg_models = {}
    sigma_main_per_regime = {}

    for regime_name in REGIME_DEFS.keys():
        df_reg = df[df["regime"] == regime_name].copy()
        if df_reg.empty:
            print(f"[train] Regime {regime_name}: no rows, skipping.")
            continue

        X_reg = df_reg[FEATURE_COLS].values
        w_reg = df_reg["sample_weight"].values

        reg_models[regime_name] = {}
        sigma_main_regime = None

        for H in HORIZONS_MINUTES:
            col = f"ret_{H}m"
            if col not in df_reg.columns:
                print(
                    f"[train] Regime {regime_name}, Horizon {H}m: "
                    f"missing {col}, skipping."
                )
                continue

            y_h = df_reg[col].values.astype("float64")
            mask = np.isfinite(y_h)
            n = int(mask.sum())
            if n < MIN_SAMPLES_PER_HORIZON:
                print(
                    f"[train] Regime {regime_name}, Horizon {H}m: "
                    f"only {n} samples, skipping."
                )
                continue

            # Regime-specific clipping for robustness
            y_h_clipped = clip_returns_for_regime(y_h, regime_name)

            print(
                f"[train] Regime {regime_name}: training regressor for {H}m on {n} samples."
            )
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
            reg.fit(X_reg[mask], y_h_clipped[mask], sample_weight=w_reg[mask])
            reg_models[regime_name][H] = reg

            # After fitting the main horizon model, compute residual std for this regime
            if H == HORIZON_MINUTES:
                y_true = y_h_clipped[mask]
                y_pred = reg.predict(X_reg[mask])
                w_h = w_reg[mask]

                if not np.any(np.isfinite(w_h)):
                    w_h = np.ones_like(y_true)

                # Normalise weights to avoid numerical issues
                w_h_norm = w_h / np.mean(w_h)

                resid = y_true - y_pred

                # Trim 5–95% quantiles to avoid a few extreme outliers
                if resid.size >= 20:
                    lo = np.percentile(resid, 5)
                    hi = np.percentile(resid, 95)
                    keep = (resid >= lo) & (resid <= hi)
                    if keep.sum() >= 5:
                        resid = resid[keep]
                        w_h_norm = w_h_norm[keep]

                mse = np.average(resid ** 2, weights=w_h_norm)
                sigma = float(np.sqrt(mse))
                sigma_main_regime = max(sigma, 1e-8)  # avoid degenerate zero

        if not reg_models[regime_name]:
            print(f"[train] Regime {regime_name}: no models trained, dropping regime.")
            reg_models.pop(regime_name, None)
            continue

        if sigma_main_regime is None:
            raise RuntimeError(
                f"No residual std computed for main horizon {HORIZON_MINUTES}m "
                f"in regime {regime_name}."
            )

        sigma_main_per_regime[regime_name] = sigma_main_regime

    if not reg_models:
        raise RuntimeError(
            "No regression models were trained for any regime/horizon."
        )

    return reg_models, sigma_main_per_regime, FEATURE_COLS


# ---------------------------------------------------------------------------
# Persist models and metadata
# ---------------------------------------------------------------------------


def save_models(s3, bucket, reg_models, sigma_main_per_regime, feature_cols):
    """
    Save:
      - reg_models (dict: regime -> horizon -> regressor) as latest_reg.pkl
      - meta.json with horizon, feature, regime, and distribution info
    """
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")

    # Serialize models into memory buffer
    buf_reg = io.BytesIO()
    joblib.dump(reg_models, buf_reg)
    buf_reg.seek(0)

    # Date-stamped key
    key_reg = f"models/quantile/{date_part}/reg_multi.pkl"
    s3.put_object(Bucket=bucket, Key=key_reg, Body=buf_reg.getvalue())

    # "Latest" pointer (what score_latest.py will load)
    s3.put_object(
        Bucket=bucket,
        Key="models/quantile/latest_reg.pkl",
        Body=buf_reg.getvalue(),
    )

    # Choose a global sigma_main for backwards compatibility (use mid if available)
    if isinstance(sigma_main_per_regime, dict) and sigma_main_per_regime:
        if "mid" in sigma_main_per_regime:
            sigma_main_global = float(sigma_main_per_regime["mid"])
        else:
            sigma_main_global = float(
                np.mean(list(sigma_main_per_regime.values()))
            )
    else:
        # Fallback: treat input as already a scalar
        sigma_main_global = float(sigma_main_per_regime)

    # Metadata
    meta = {
        "horizon_minutes": HORIZON_MINUTES,          # main horizon (60m)
        "path_horizons_minutes": HORIZONS_MINUTES,   # [5,10,...,120]
        "window_days": WINDOW_DAYS,
        "decay_days": DECAY_DAYS,
        "feature_cols": feature_cols,                # used for regressors
        "tax_rate": TAX_RATE,
        "margin": MARGIN,
        "sigma_main": float(sigma_main_global),
        "sigma_main_per_regime": {
            k: float(v) for k, v in sigma_main_per_regime.items()
        },
        "regime_defs": REGIME_DEFS,
        "generated_at_iso": now.isoformat(),
    }

    meta_bytes = json_bytes(meta)
    key_meta = f"models/quantile/{date_part}/meta.json"
    s3.put_object(Bucket=bucket, Key=key_meta, Body=meta_bytes)
    s3.put_object(
        Bucket=bucket,
        Key="models/quantile/latest_meta.json",
        Body=meta_bytes,
    )


def json_bytes(obj):
    import json
    return json.dumps(obj).encode("utf-8")


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

    reg_models, sigma_main_per_regime, feature_cols = train_models(df)
    save_models(s3, bucket, reg_models, sigma_main_per_regime, feature_cols)
    print("Training complete.")
    print("Main-horizon residual sigma by regime:")
    for name, sigma in sigma_main_per_regime.items():
        print(f"  - {name}: {sigma:.6f}")


if __name__ == "__main__":
    main()
