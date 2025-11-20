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

EXPERIMENT = "quantile"

# ---------------------------------------------------------------------------
# Config (constants should match your training config)
# ---------------------------------------------------------------------------

HORIZON_MINUTES = 60
TAX_RATE = 0.02
MARGIN = 0.002

# How many days of 5m snapshots to use for evaluation
EVAL_DAYS = 14

# Recency decay for sample weights (same as training)
DECAY_DAYS = 14

# Minimum rows for a bucket combination to be included in the detailed output
MIN_ROWS_PER_BUCKET = 100

# Default regime definitions (used if meta has none)
REGIME_DEFS_DEFAULT = {
    "low":  {"mid_price_min": 0,       "mid_price_max": 10_000},
    "mid":  {"mid_price_min": 10_000,  "mid_price_max": 100_000},
    "high": {"mid_price_min": 100_000, "mid_price_max": None},
}


# ---------------------------------------------------------------------------
# Helpers to load data and models
# ---------------------------------------------------------------------------

def assign_regime_for_price(mid_price: float, regime_defs) -> str:
    """
    Assign a regime name given a mid_price and a regime_defs dict.
    Falls back to 'mid' if mid_price is missing or doesn't match.
    """
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
      - future_return (60m actual return)
      - sample_weight (recency-based)
    """
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=EVAL_DAYS - 1)).date()
    end_date = now.date()  # include today

    chunks = []
    d = start_date
    while d <= end_date:
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        print("[load]", d, "keys:", len(keys))
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
    df["future_return"] = df[col_60]

    # Recency-based sample weights (same as training)
    latest_ts = df["timestamp"].max()
    age_days = (latest_ts - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["sample_weight"] = np.exp(-age_days / DECAY_DAYS)

    # Keep only rows with reasonable mid_price
    df = df[df["mid_price"] > 0].copy()

    print(f"[load] Eval dataframe rows after cleaning: {len(df)}")
    return df


def load_latest_regressor(s3, bucket: str):
    """
    Load latest regression models + meta from R2.

    Returns:
      - reg_main_map: dict[regime_name] -> main-horizon regressor
      - sigma_main_per_regime: dict[regime_name] -> float
      - feature_cols: list of feature columns
      - regime_defs: dict used for regime assignment
    """
    key_reg = "models/quantile/latest_reg.pkl"
    key_meta = "models/quantile/latest_meta.json"

    obj_reg = s3.get_object(Bucket=bucket, Key=key_reg)
    obj_meta = s3.get_object(Bucket=bucket, Key=key_meta)

    reg_models_raw = joblib.load(io.BytesIO(obj_reg["Body"].read()))
    meta = pd.read_json(io.BytesIO(obj_meta["Body"].read()), typ="series").to_dict()

    sigma_main_global = float(meta.get("sigma_main", 0.02))
    sigma_main_per_regime_meta = meta.get("sigma_main_per_regime")
    regime_defs_meta = meta.get("regime_defs")
    feature_cols = meta.get(
        "feature_cols",
        ["mid_price", "spread_pct", "log_volume_5m"],
    )

    if not reg_models_raw:
        raise RuntimeError("reg_models is empty in latest_reg.pkl")

    first_key = next(iter(reg_models_raw.keys()))
    if isinstance(first_key, int):
        # Old format: single global dict[horizon] -> regressor
        if HORIZON_MINUTES not in reg_models_raw:
            raise RuntimeError(
                f"No regressor for main horizon {HORIZON_MINUTES}m in latest_reg.pkl"
            )
        reg_main_map = {"global": reg_models_raw[HORIZON_MINUTES]}
        regime_defs = regime_defs_meta or {"global": {"mid_price_min": 0, "mid_price_max": None}}
        if isinstance(sigma_main_per_regime_meta, dict) and sigma_main_per_regime_meta:
            sigma_main_per_regime = {k: float(v) for k, v in sigma_main_per_regime_meta.items()}
        else:
            sigma_main_per_regime = {"global": sigma_main_global}
    else:
        # New format: regime_name -> {horizon: regressor}
        reg_main_map = {}
        for regime_name, models_for_regime in reg_models_raw.items():
            if HORIZON_MINUTES in models_for_regime:
                reg_main_map[regime_name] = models_for_regime[HORIZON_MINUTES]
        if not reg_main_map:
            raise RuntimeError(
                f"No regressor for main horizon {HORIZON_MINUTES}m in any regime."
            )
        regime_defs = regime_defs_meta or REGIME_DEFS_DEFAULT
        if isinstance(sigma_main_per_regime_meta, dict) and sigma_main_per_regime_meta:
            sigma_main_per_regime = {k: float(v) for k, v in sigma_main_per_regime_meta.items()}
        else:
            sigma_main_per_regime = {
                regime_name: sigma_main_global for regime_name in reg_main_map.keys()
            }

    return reg_main_map, sigma_main_per_regime, feature_cols, regime_defs


def normal_cdf_array(z: np.ndarray) -> np.ndarray:
    """
    Vectorised Normal(0,1) CDF using math.erf.
    """
    def _cdf_scalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    vec = np.vectorize(_cdf_scalar, otypes=[float])
    return vec(z)


def add_predictions_and_residuals(
    df: pd.DataFrame,
    reg_main_map,
    sigma_main_per_regime: dict,
    feature_cols,
    regime_defs,
) -> pd.DataFrame:
    """
    Given a dataframe with features and future_return, add:

      - regime (based on mid_price)
      - future_return_hat
      - net_return_hat (after tax)
      - prob_profit (Win%)
      - resid = future_return - future_return_hat
    """
    df = df.copy()

    # Assign regimes
    df["regime"] = df["mid_price"].apply(
        lambda p: assign_regime_for_price(p, regime_defs)
    )

    n_rows = len(df)
    X_all = df[feature_cols].values

    future_return_hat = np.full(n_rows, np.nan, dtype="float64")
    regime_sigma = np.full(n_rows, np.nan, dtype="float64")

    # Predict per regime
    for regime_name, reg in reg_main_map.items():
        mask = df["regime"] == regime_name
        if not mask.any():
            continue
        X_reg = X_all[mask.values]
        y_hat = reg.predict(X_reg)
        future_return_hat[mask.values] = y_hat

        sigma = sigma_main_per_regime.get(regime_name)
        if sigma is None:
            # fallback to global median
            sigma_vals = np.array(list(sigma_main_per_regime.values()), dtype="float64")
            sigma_vals = sigma_vals[np.isfinite(sigma_vals) & (sigma_vals > 0)]
            if sigma_vals.size > 0:
                sigma = float(np.median(sigma_vals))
            else:
                sigma = 0.02
        regime_sigma[mask.values] = float(sigma)

    df["future_return_hat"] = future_return_hat
    df["regime_sigma"] = regime_sigma
    df["net_return_hat"] = (1.0 + df["future_return_hat"]) * (1.0 - TAX_RATE) - 1.0

    # Profit_ok is defined on net_return > MARGIN; translate to gross threshold
    gross_threshold = (1.0 + MARGIN) / (1.0 - TAX_RATE) - 1.0

    sigma_vec = df["regime_sigma"].values
    sigma_vec = np.where(
        (np.isfinite(sigma_vec)) & (sigma_vec > 0.0),
        sigma_vec,
        0.02,
    )
    z = (df["future_return_hat"].values - gross_threshold) / np.maximum(sigma_vec, 1e-8)
    prob_profit = normal_cdf_array(z)
    df["prob_profit"] = np.clip(prob_profit, 1e-4, 1.0 - 1e-4)

    df["resid"] = df["future_return"] - df["future_return_hat"]

    return df


# ---------------------------------------------------------------------------
# Bucketing / sigma computation
# ---------------------------------------------------------------------------

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
    hours = ts.dt.hour + ts.dt.minute / 60.0

    # 4-hour ranges: [0-4), [4-8), ..., [20-24)
    time_edges = [0, 4, 8, 12, 16, 20, 24]
    time_labels = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    df["time_bin"] = pd.cut(
        hours,
        bins=time_edges,
        labels=time_labels,
        right=False,
        include_lowest=True,
    )

    # Price ranges (in gp), matching your previous analysis
    price_edges = [-np.inf, 1_000, 10_000, 100_000, 1_000_000, np.inf]
    price_labels = ["<=1k", "1k-10k", "10k-100k", "100k-1m", ">1m"]
    df["price_bin"] = pd.cut(
        df["mid_price"],
        bins=price_edges,
        labels=price_labels,
        right=False,
        include_lowest=True,
    )

    # Predicted Win% bins
    win_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    win_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df["win_bin"] = pd.cut(
        df["prob_profit"],
        bins=win_edges,
        labels=win_labels,
        right=False,
        include_lowest=True,
    )

    # Predicted net return bins (profit/loss%)
    # net_return_hat is in decimal form (e.g. 0.05 = +5%)
    ret_edges = [-1.0, -0.05, 0.0, 0.05, 0.2, np.inf]
    ret_labels = ["<-5%", "-5-0%", "0-5%", "5-20%", ">20%"]
    df["pred_ret_bin"] = pd.cut(
        df["net_return_hat"],
        bins=ret_edges,
        labels=ret_labels,
        right=False,
        include_lowest=True,
    )

    return df


def compute_bucket_sigmas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute residual sigma (unweighted and weighted) for each combination of:
      (regime, time_bin, price_bin, win_bin, pred_ret_bin)

    Returns a DataFrame with one row per non-empty bucket combo.
    """
    group_cols = ["regime", "time_bin", "price_bin", "win_bin", "pred_ret_bin"]

    rows = []
    for key, sub in df.groupby(group_cols, dropna=True):
        n = len(sub)
        if n < MIN_ROWS_PER_BUCKET:
            continue

        resid = sub["resid"].values
        if resid.size == 0:
            continue

        # Unweighted sigma
        sigma_unweighted = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0

        # Weighted sigma (based on sample_weight)
        w = sub["sample_weight"].values
        if not np.any(np.isfinite(w)):
            w = np.ones_like(resid)
        w = w / np.mean(w)
        mse = np.average(resid ** 2, weights=w)
        sigma_weighted = float(np.sqrt(mse))

        rows.append(
            {
                "regime": key[0],
                "time_bin": key[1],
                "price_bin": key[2],
                "win_bin": key[3],
                "pred_ret_bin": key[4],
                "n_rows": int(n),
                "sigma_unweighted": sigma_unweighted,
                "sigma_weighted": sigma_weighted,
                "mean_future_return": float(sub["future_return"].mean()),
                "mean_pred_return": float(sub["future_return_hat"].mean()),
                "mean_prob_profit": float(sub["prob_profit"].mean()),
                "mean_pred_net_return": float(sub["net_return_hat"].mean()),
            }
        )

    if not rows:
        return pd.DataFrame()

    df_buckets = pd.DataFrame(rows)
    df_buckets.sort_values(
        ["sigma_unweighted", "n_rows"],
        ascending=[False, False],
        inplace=True,
    )
    return df_buckets


def print_marginal_summaries(df_buckets: pd.DataFrame):
    """
    Print marginal sigma summaries by each bin dimension separately
    (this is what you'll paste back into ChatGPT).
    """
    def summarize(col):
        print("\n=== MARGINAL by", col, "===")
        grp = df_buckets.groupby(col).agg(
            n_rows=("n_rows", "sum"),
            sigma_unweighted=("sigma_unweighted", "mean"),
            sigma_weighted=("sigma_weighted", "mean"),
        )
        grp = grp.sort_index()
        print(grp.to_string())

    for col in ["regime", "time_bin", "price_bin", "win_bin", "pred_ret_bin"]:
        if col in df_buckets.columns:
            summarize(col)


def compute_regime_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall sigma per regime (unweighted and weighted) for the main horizon.
    """
    rows = []
    for regime_name, sub in df.groupby("regime"):
        n = len(sub)
        resid = sub["resid"].values
        if n < MIN_ROWS_PER_BUCKET or resid.size == 0:
            continue

        sigma_unweighted = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0

        w = sub["sample_weight"].values
        if not np.any(np.isfinite(w)):
            w = np.ones_like(resid)
        w = w / np.mean(w)
        mse = np.average(resid ** 2, weights=w)
        sigma_weighted = float(np.sqrt(mse))

        rows.append(
            {
                "regime": regime_name,
                "n_rows": int(n),
                "sigma_unweighted": sigma_unweighted,
                "sigma_weighted": sigma_weighted,
                "mean_future_return": float(sub["future_return"].mean()),
                "mean_pred_return": float(sub["future_return_hat"].mean()),
                "mean_prob_profit": float(sub["prob_profit"].mean()),
                "mean_pred_net_return": float(sub["net_return_hat"].mean()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "regime",
            "n_rows",
            "sigma_unweighted",
            "sigma_weighted",
            "mean_future_return",
            "mean_pred_return",
            "mean_prob_profit",
            "mean_pred_net_return",
        ])

    df_reg = pd.DataFrame(rows)
    df_reg.sort_values("sigma_unweighted", ascending=False, inplace=True)
    return df_reg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()
    local_path = "sigma_buckets_full.csv"

    def write_and_push(out_df: pd.DataFrame, reason: str = ""):
        """Write CSV locally (always) and push to R2 (unless no prefix)."""
        out_df.to_csv(local_path, index=False)
        print(f"[save] Wrote bucket CSV locally to {local_path} (rows={len(out_df)}){(' - ' + reason) if reason else ''}")

        prefix = os.environ.get("SIGMA_BUCKET_PREFIX") or os.environ.get("SIGMA_PREFIX")
        if not prefix:
            branch = os.environ.get("GITHUB_REF_NAME", "").lower()
            if "remove-noisy-sections" in branch:
                prefix = "analysis/remove-noisy-sections"
            elif "quantile" in branch:
                prefix = "analysis/quantile"
            elif branch:
                prefix = f"analysis/{branch}"
            else:
                prefix = "analysis/default"
        prefix = prefix.rstrip("/")

        now = datetime.now(timezone.utc)
        key = f"{prefix}/sigma_buckets_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        buf = io.BytesIO(out_df.to_csv(index=False).encode("utf-8"))
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        print(f"[save] Wrote bucket CSV to r2://{bucket}/{key}")

    empty_cols = [
        "time_bin", "price_bin", "win_bin", "pred_ret_bin",
        "n_rows", "sigma_unweighted", "sigma_weighted",
        "mean_future_return", "mean_pred_return",
        "mean_prob_profit", "mean_pred_net_return",
    ]

    # 1) Load eval data
    df = build_eval_dataframe(s3, bucket)
    if df.empty:
        write_and_push(pd.DataFrame(columns=empty_cols), reason="no eval data")
        # Still write an empty regime summary for completeness
        pd.DataFrame(columns=["regime", "n_rows", "sigma_unweighted", "sigma_weighted"]).to_csv(
            "sigma_regimes.csv", index=False
        )
        return

    # 2) Load regressors and meta
    reg_main_map, sigma_main_per_regime, feature_cols, regime_defs = load_latest_regressor(s3, bucket)
    print(f"[main] Loaded main-horizon regressors for regimes: {list(reg_main_map.keys())}")
    print(f"[main] sigma_main_per_regime: {sigma_main_per_regime}")
    print(f"[main] Using {len(feature_cols)} features:", feature_cols)

    # 3) Add predictions, residuals, Win%, regimes
    df = add_predictions_and_residuals(df, reg_main_map, sigma_main_per_regime, feature_cols, regime_defs)

    # 4) Add bins (time, price, Win%, predicted return)
    df = add_bins(df)

    # 5) Compute bucket sigmas
    df_buckets = compute_bucket_sigmas(df)

    if df_buckets.empty:
        write_and_push(pd.DataFrame(columns=empty_cols), reason="no buckets met min rows")
        df_reg = compute_regime_summaries(df)
        df_reg.to_csv("sigma_regimes.csv", index=False)
        print(f"[main] Wrote per-regime summary to sigma_regimes.csv (rows: {len(df_reg)})")
        return

    # Save a full CSV so we can paste summaries externally if needed
    out_csv = df_buckets.copy()
    out_csv["sigma_unweighted"] = out_csv["sigma_unweighted"].round(3)
    out_csv["sigma_weighted"] = out_csv["sigma_weighted"].round(3)
    out_csv["mean_future_return"] = out_csv["mean_future_return"].round(3)
    out_csv["mean_pred_return"] = out_csv["mean_pred_return"].round(3)
    out_csv["mean_prob_profit"] = out_csv["mean_prob_profit"].round(3)
    out_csv["mean_pred_net_return"] = out_csv["mean_pred_net_return"].round(3)

    write_and_push(out_csv)

    # Compute and save per-regime summary
    df_reg = compute_regime_summaries(df)
    df_reg.to_csv("sigma_regimes.csv", index=False)
    print(f"[main] Wrote per-regime summary to sigma_regimes.csv:")
    print(df_reg.to_string(index=False))

    # Print marginal summaries (small enough to paste into chat)
    print_marginal_summaries(df_buckets)

if __name__ == "__main__":
    main()
