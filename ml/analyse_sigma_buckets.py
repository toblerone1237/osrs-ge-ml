import os
import io
from datetime import datetime, timedelta, timezone
from math import erf, sqrt

import numpy as np
import pandas as pd
import joblib

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
    add_model_features,
    add_multi_horizon_returns,
    HORIZONS_MINUTES,
)

# Match training config
HORIZON_MINUTES = 60
TAX_RATE = 0.02
MARGIN = 0.002

# How many days of history to analyse
EVAL_DAYS = 14

# Recency decay for sample weights (same as training)
DECAY_DAYS = 14

# Minimum rows for a bucket combination to be included in the detailed output
MIN_ROWS_PER_BUCKET = 100


# ---------------------------------------------------------------------------
# Helpers to load data and models
# ---------------------------------------------------------------------------

def build_eval_dataframe(s3, bucket: str) -> pd.DataFrame:
    """
    Load 5m snapshots for the last EVAL_DAYS days, add model features and
    60m future returns.

    Returns a DataFrame with:
      - timestamp (UTC)
      - timestamp_unix
      - item_id
      - mid_price
      - model features
      - future_return (60m actual return, UNCLIPPED on purpose)
      - sample_weight (recency-based)
    """
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=EVAL_DAYS - 1)).date()
    end_date = now.date()

    print(f"[load] Building eval dataframe from {start_date} to {end_date}")

    chunks = []
    d = start_date
    while d <= end_date:
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        print(f"[load] {d}: {len(keys)} snapshot keys")
        if keys:
            df_day = flatten_5m_snapshots(s3, bucket, keys)
            if not df_day.empty:
                chunks.append(df_day)
        d += timedelta(days=1)

    if not chunks:
        print("[load] No 5m data found for eval window.")
        return pd.DataFrame()

    df_raw = pd.concat(chunks, ignore_index=True)
    # Add model features
    df = add_model_features(df_raw)

    # Add only the main horizon returns (we don't need full path)
    df = add_multi_horizon_returns(df, horizons_minutes=[HORIZON_MINUTES])

    col_60 = f"ret_{HORIZON_MINUTES}m"
    if col_60 not in df.columns:
        raise RuntimeError(f"Expected column {col_60} from add_multi_horizon_returns().")

    df = df.dropna(subset=[col_60]).copy()
    df["future_return"] = df[col_60]  # UNCLIPPED (so we see the true tails)

    # Recency-based sample weights (same as training)
    latest_ts = df["timestamp"].max()
    age_days = (latest_ts - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["sample_weight"] = np.exp(-age_days / DECAY_DAYS)

    # Keep only rows with positive mid_price
    df = df[df["mid_price"] > 0].copy()

    print(f"[load] Eval dataframe rows after basic cleaning: {len(df)}")
    return df


def load_latest_regressor(s3, bucket):
    key_reg = "models/remove-noisy-sections/xgb/latest_reg.pkl"
    key_meta = "models/remove-noisy-sections/xgb/latest_meta.json"

    obj_reg = s3.get_object(Bucket=bucket, Key=key_reg)
    obj_meta = s3.get_object(Bucket=bucket, Key=key_meta)

    reg_models = joblib.load(io.BytesIO(obj_reg["Body"].read()))
    meta = pd.read_json(io.BytesIO(obj_meta["Body"].read()), typ="series").to_dict()

    horizon_minutes = int(meta.get("horizon_minutes", HORIZON_MINUTES))
    sigma_main = float(meta.get("sigma_main", 0.02))
    feature_cols = meta.get("feature_cols", ["mid_price", "spread_pct", "log_volume_5m"])

    if horizon_minutes not in reg_models:
        raise RuntimeError(f"No regressor for horizon {horizon_minutes}m.")

    reg_main = reg_models[horizon_minutes]
    return reg_main, sigma_main, feature_cols


def normal_cdf_array(z: np.ndarray) -> np.ndarray:
    def _cdf_scalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return np.vectorize(_cdf_scalar, otypes=[float])(z)


def add_predictions_and_residuals(df: pd.DataFrame, reg_main, sigma_main: float, feature_cols):
    """
    Given a dataframe with features and UNCLIPPED future_return, add:

      - future_return_hat (from the trained model)
      - net_return_hat (after tax)
      - prob_profit (Win%) via Normal CDF using sigma_main
      - resid = future_return - future_return_hat
    """
    df = df.copy()

    X = df[feature_cols].values
    future_return_hat = reg_main.predict(X)

    df["future_return_hat"] = future_return_hat
    df["net_return_hat"] = (1.0 + df["future_return_hat"]) * (1.0 - TAX_RATE) - 1.0

    # Profit_ok is defined on net_return > MARGIN; translate to gross threshold
    gross_threshold = (1.0 + MARGIN) / (1.0 - TAX_RATE) - 1.0
    z = (df["future_return_hat"].values - gross_threshold) / max(sigma_main, 1e-8)
    prob_profit = normal_cdf_array(z)
    df["prob_profit"] = np.clip(prob_profit, 1e-4, 1.0 - 1e-4)

    df["resid"] = df["future_return"] - df["future_return_hat"]
    return df


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bin columns:

      - time_bin: hour-of-day buckets (4h ranges)
      - price_bin: mid_price ranges
      - win_bin: predicted Win% ranges
      - pred_ret_bin: predicted net return (profit/loss%) ranges
    """
    df = df.copy()

    # Time-of-day in hours (0-24)
    ts = df["timestamp"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(timezone.utc)
    else:
        ts = ts.dt.tz_localize(timezone.utc)

    hour_float = ts.dt.hour + ts.dt.minute / 60.0

    time_edges = [0, 4, 8, 12, 16, 20, 24]
    time_labels = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    df["time_bin"] = pd.cut(hour_float, bins=time_edges, labels=time_labels, right=False, include_lowest=True)

    # Price bins (in gp)
    price_edges = [0, 1e3, 1e4, 1e5, 1e6, np.inf]
    price_labels = ["<=1k", "1k-10k", "10k-100k", "100k-1m", ">1m"]
    df["price_bin"] = pd.cut(df["mid_price"], bins=price_edges, labels=price_labels, right=False, include_lowest=True)

    # Win% bins
    win_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    win_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df["win_bin"] = pd.cut(df["prob_profit"], bins=win_edges, labels=win_labels, right=False, include_lowest=True)

    # Predicted net return bins (decimal, e.g. 0.05=+5%)
    ret_edges = [-1.0, -0.05, 0.0, 0.05, 0.2, np.inf]
    ret_labels = ["<-5%", "-5-0%", "0-5%", "5-20%", ">20%"]
    df["pred_ret_bin"] = pd.cut(df["net_return_hat"], bins=ret_edges, labels=ret_labels, right=False, include_lowest=True)

    return df


def compute_bucket_sigmas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute residual sigma (unweighted and weighted) for each combination of:
      (time_bin, price_bin, win_bin, pred_ret_bin)
    """
    group_cols = ["time_bin", "price_bin", "win_bin", "pred_ret_bin"]

    rows = []
    for key, sub in df.groupby(group_cols, dropna=True):
        n = len(sub)
        if n < MIN_ROWS_PER_BUCKET:
            continue

        resid = sub["resid"].values
        w = sub["sample_weight"].values

        sigma_unw = float(np.std(resid, ddof=0))
        if w.sum() > 0:
            sigma_w = float(np.sqrt(np.average(resid**2, weights=w)))
        else:
            sigma_w = float("nan")

        rows.append(
            {
                "time_bin": str(key[0]),
                "price_bin": str(key[1]),
                "win_bin": str(key[2]),
                "pred_ret_bin": str(key[3]),
                "n_rows": int(n),
                "sigma_unweighted": sigma_unw,
                "sigma_weighted": sigma_w,
                "mean_future_return": float(sub["future_return"].mean()),
                "mean_pred_return": float(sub["future_return_hat"].mean()),
                "mean_prob_profit": float(sub["prob_profit"].mean()),
                "mean_pred_net_return": float(sub["net_return_hat"].mean()),
            }
        )

    if not rows:
        return pd.DataFrame()

    df_buckets = pd.DataFrame(rows)
    df_buckets.sort_values(["sigma_unweighted", "n_rows"], ascending=[False, False], inplace=True)
    return df_buckets


def print_marginal_summaries(df_buckets: pd.DataFrame):
    def summarize(col):
        print("\n=== MARGINAL by", col, "===")
        grp = df_buckets.groupby(col).agg(
            n_rows=("n_rows", "sum"),
            sigma_unweighted=("sigma_unweighted", "mean"),
            sigma_weighted=("sigma_weighted", "mean"),
        )
        grp = grp.sort_index()
        print(grp.to_string())

    for col in ["time_bin", "price_bin", "win_bin", "pred_ret_bin"]:
        summarize(col)


def summarize():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    df = build_eval_dataframe(s3, bucket)
    if df.empty:
        print("No data to evaluate.")
        return

    reg_main, sigma_main, feature_cols = load_latest_regressor(s3, bucket)
    print(f"[meta] Loaded main regressor; sigma_main={sigma_main:.6f}")

    df = add_predictions_and_residuals(df, reg_main, sigma_main, feature_cols)
    df = add_bins(df)
    df_buckets = compute_bucket_sigmas(df)

    print("\n=== TOP NOISY BUCKETS (by unweighted sigma) ===")
    print(df_buckets.head(20).to_string())

    print_marginal_summaries(df_buckets)

    # Save a full CSV so we can paste summaries externally if needed
    out_csv = df_buckets.copy()
    out_csv["sigma_unweighted"] = out_csv["sigma_unweighted"].round(3)
    out_csv["sigma_weighted"] = out_csv["sigma_weighted"].round(3)
    out_csv["mean_future_return"] = out_csv["mean_future_return"].round(3)
    out_csv["mean_pred_return"] = out_csv["mean_pred_return"].round(3)
    out_csv["mean_prob_profit"] = out_csv["mean_prob_profit"].round(3)
    out_csv["mean_pred_net_return"] = out_csv["mean_pred_net_return"].round(3)

    # Write locally so the workflow can upload an artifact / push to R2
    local_path = "sigma_buckets_full.csv"
    out_csv.to_csv(local_path, index=False)
    print(f"[save] Wrote bucket CSV locally to {local_path} (rows={len(out_csv)})")

    prefix = (
        os.environ.get("SIGMA_BUCKET_PREFIX")
        or os.environ.get("SIGMA_PREFIX")
        or "analysis/remove-noisy-sections"
    )
    prefix = prefix.rstrip("/")
    now = datetime.now(timezone.utc)
    key = f"{prefix}/sigma_buckets_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    buf = io.BytesIO(out_csv.to_csv(index=False).encode("utf-8"))
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"[save] Wrote bucket CSV to r2://{bucket}/{key}")


def main():
    summarize()


if __name__ == "__main__":
    main()
