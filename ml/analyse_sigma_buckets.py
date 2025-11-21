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
    compute_return_scale,
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

# Controls for permutation importance (kept modest to avoid huge runtime)
PERMUTATION_SAMPLE_ROWS = int(os.environ.get("PERMUTATION_SAMPLE_ROWS", "8000"))
PERMUTATION_RANDOM_SEED = int(os.environ.get("PERMUTATION_RANDOM_SEED", "1337"))

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
      - future_return (60m actual return, UNCLIPPED on purpose)
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
    df["future_return"] = df[col_60]  # UNCLIPPED (so we see the true tails)

    # Recency-based sample weights (same as training)
    latest_ts = df["timestamp"].max()
    age_days = (latest_ts - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["sample_weight"] = np.exp(-age_days / DECAY_DAYS)

    # Keep only rows with positive mid_price
    df = df[df["mid_price"] > 0].copy()

    print(f"[load] Eval dataframe rows after basic cleaning: {len(df)}")
    return df


def load_latest_regressor(s3, bucket: str):
    """
    Load latest regression models + meta from R2.

    Returns:
      - reg_main_map: dict[regime_name] -> main-horizon regressor
      - sigma_main_per_regime: dict[regime_name] -> float
      - feature_cols: list of feature columns
      - regime_defs: dict used for regime assignment
      - return_scaling_cfg: dict describing return scaling (if any)
      - calibration_main_per_regime: dict describing per-regime calibration (if any)
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
    return_scaling_cfg = meta.get("return_scaling") or {}
    calibration_main_per_regime = meta.get("calibration_main_per_regime") or {}
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

    return (
        reg_main_map,
        sigma_main_per_regime,
        feature_cols,
        regime_defs,
        return_scaling_cfg,
        calibration_main_per_regime,
    )


def normal_cdf_array(z: np.ndarray) -> np.ndarray:
    def _cdf_scalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    vec = np.vectorize(_cdf_scalar, otypes=[float])
    return vec(z)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _safe_weights(w: np.ndarray, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n, dtype="float64")
    w = np.asarray(w, dtype="float64")
    if w.shape[0] != n:
        return np.ones(n, dtype="float64")
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if not np.any(w):
        w = np.ones(n, dtype="float64")
    # normalise to keep scales comparable
    return w / np.mean(w)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    w = _safe_weights(w, len(x))
    x = np.asarray(x, dtype="float64")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.average(x[mask], weights=w[mask]))


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    w = _safe_weights(w, len(y_true))
    err = y_true - y_pred
    err = np.where(np.isfinite(err), err, 0.0)
    return float(np.average(err ** 2, weights=w))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    w = _safe_weights(w, len(y_true))
    err = np.abs(y_true - y_pred)
    err = np.where(np.isfinite(err), err, 0.0)
    return float(np.average(err, weights=w))


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    w = _safe_weights(w, len(x))[mask]
    wx = np.average(x, weights=w)
    wy = np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    var_x = np.average((x - wx) ** 2, weights=w)
    var_y = np.average((y - wy) ** 2, weights=w)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return float(cov / np.sqrt(var_x * var_y))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    w = _safe_weights(w, len(y_true))
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    w = w[mask]
    y_bar = np.average(y_true, weights=w)
    sst = np.average((y_true - y_bar) ** 2, weights=w)
    if sst <= 0:
        return float("nan")
    sse = np.average((y_true - y_pred) ** 2, weights=w)
    return float(1.0 - sse / sst)


def residual_quantiles(resid: np.ndarray, quantiles=None) -> dict:
    if quantiles is None:
        quantiles = [1, 5, 25, 50, 75, 95, 99]
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return {f"p{q}": float("nan") for q in quantiles}
    return {f"p{q}": float(np.percentile(resid, q)) for q in quantiles}


def calibration_stats(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> dict:
    w = _safe_weights(w, len(y_true))
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"slope": float("nan"), "intercept": float("nan")}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    w = w[mask]
    x_bar = np.average(y_pred, weights=w)
    y_bar = np.average(y_true, weights=w)
    var_x = np.average((y_pred - x_bar) ** 2, weights=w)
    if var_x <= 1e-12:
        return {"slope": 1.0, "intercept": 0.0}
    cov_xy = np.average((y_pred - x_bar) * (y_true - y_bar), weights=w)
    slope = cov_xy / var_x
    intercept = y_bar - slope * x_bar
    return {"slope": float(slope), "intercept": float(intercept)}


def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall regression metrics (weighted + unweighted), residual stats, and calibration.
    """
    y_true_raw = df["future_return"].to_numpy(dtype="float64")
    y_pred_raw = df["future_return_hat"].to_numpy(dtype="float64")
    w_raw = df["sample_weight"].to_numpy(dtype="float64")

    mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)
    if not np.any(mask):
        return pd.DataFrame()

    y_true = y_true_raw[mask]
    y_pred = y_pred_raw[mask]
    w = w_raw[mask]
    resid = y_true - y_pred

    mse_w = weighted_mse(y_true, y_pred, w)
    mae_w = weighted_mae(y_true, y_pred, w)
    r2_w = weighted_r2(y_true, y_pred, w)
    corr_w = weighted_corr(y_pred, y_true, w)

    mse = float(np.mean((resid) ** 2))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1.0 - mse / np.var(y_true)) if np.var(y_true) > 0 else float("nan")
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else float("nan")

    calib = calibration_stats(y_true, y_pred, w)
    resid_q = residual_quantiles(resid)

    net_true = (1.0 + y_true) * (1.0 - TAX_RATE) - 1.0
    net_pred = df.loc[mask, "net_return_hat"].to_numpy(dtype="float64")

    out = {
        "n_rows": int(len(y_true)),
        "mean_future_return": float(np.mean(y_true)),
        "mean_pred_return": float(np.mean(y_pred)),
        "mean_net_true": float(np.mean(net_true)),
        "mean_net_pred": float(np.mean(net_pred)),
        "rmse": float(np.sqrt(mse)),
        "rmse_weighted": float(np.sqrt(mse_w)),
        "mae": mae,
        "mae_weighted": mae_w,
        "r2": r2,
        "r2_weighted": r2_w,
        "corr": corr,
        "corr_weighted": corr_w,
        "calibration_slope": calib["slope"],
        "calibration_intercept": calib["intercept"],
    }
    out.update({f"resid_{k}": v for k, v in resid_q.items()})
    return pd.DataFrame([out])


def compute_prob_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classification-like metrics on profit prediction (using prob_profit and profit_ok).
    """
    if "profit_ok" not in df.columns:
        return pd.DataFrame()

    y_true = df["profit_ok"].to_numpy(dtype="int32")
    prob = df["prob_profit"].to_numpy(dtype="float64")
    w = df["sample_weight"].to_numpy(dtype="float64")

    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    w_safe = _safe_weights(w, len(prob))

    brier = float(np.average((prob - y_true) ** 2, weights=w_safe))
    logloss = float(
        -np.average(
            y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob),
            weights=w_safe,
        )
    )

    base_rate = float(np.mean(y_true)) if len(y_true) else float("nan")
    pred_rate = float(np.mean(prob))

    def threshold_stats(thresh: float):
        mask = prob >= thresh
        if not np.any(mask):
            return {"n": 0, "hit_rate": float("nan"), "avg_net": float("nan")}
        hits = y_true[mask]
        net = df.loc[mask, "net_return"].to_numpy(dtype="float64")
        return {
            "n": int(mask.sum()),
            "hit_rate": float(np.mean(hits)),
            "avg_net": float(np.mean(net)),
        }

    t0 = threshold_stats(0.5)
    t60 = threshold_stats(0.6)

    return pd.DataFrame(
        [
            {
                "brier_weighted": brier,
                "logloss_weighted": logloss,
                "base_rate_profit": base_rate,
                "mean_pred_prob": pred_rate,
                "p0_5_n": t0["n"],
                "p0_5_hit_rate": t0["hit_rate"],
                "p0_5_avg_net": t0["avg_net"],
                "p0_6_n": t60["n"],
                "p0_6_hit_rate": t60["hit_rate"],
                "p0_6_avg_net": t60["avg_net"],
            }
        ]
    )


def compute_feature_correlations(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Univariate correlation of each feature with future_return and residuals.
    """
    rows = []
    y_true = df["future_return"].to_numpy(dtype="float64")
    resid = (df["future_return"] - df["future_return_hat"]).to_numpy(dtype="float64")
    w = df["sample_weight"].to_numpy(dtype="float64")

    for feat in feature_cols:
        if feat not in df.columns:
            continue
        x = df[feat].to_numpy(dtype="float64")
        mask = np.isfinite(x) & np.isfinite(y_true)
        n = int(mask.sum())
        if n < 10:
            continue
        corr_y = weighted_corr(x[mask], y_true[mask], w[mask])
        corr_resid = weighted_corr(x[mask], resid[mask], w[mask])
        rows.append(
            {
                "feature": feat,
                "n": n,
                "mean": float(np.mean(x[mask])),
                "std": float(np.std(x[mask])) if n > 1 else 0.0,
                "corr_future": corr_y,
                "corr_residual": corr_resid,
            }
        )

    return pd.DataFrame(rows).sort_values("corr_residual", ascending=False)


def compute_permutation_importance(
    df: pd.DataFrame,
    reg_main_map,
    sigma_main_per_regime,
    feature_cols,
    regime_defs,
    use_return_scaling: bool,
    calibration_main_per_regime: dict,
    sample_rows: int = PERMUTATION_SAMPLE_ROWS,
) -> pd.DataFrame:
    """
    Permutation importance on a sampled subset using weighted MSE on future_return.
    """
    if len(df) == 0 or sample_rows <= 0:
        return pd.DataFrame()

    n_sample = min(len(df), sample_rows)
    sample = df.sample(n=n_sample, random_state=PERMUTATION_RANDOM_SEED).copy()

    def predict_and_score(sub: pd.DataFrame) -> float:
        tmp = add_predictions_and_residuals(
            sub,
            reg_main_map,
            sigma_main_per_regime,
            feature_cols,
            regime_defs,
            use_return_scaling,
            calibration_main_per_regime,
        )
        return weighted_mse(
            tmp["future_return"].to_numpy(dtype="float64"),
            tmp["future_return_hat"].to_numpy(dtype="float64"),
            tmp["sample_weight"].to_numpy(dtype="float64"),
        )

    base_mse = predict_and_score(sample)
    rows = []

    rng = np.random.default_rng(PERMUTATION_RANDOM_SEED)
    for feat in feature_cols:
        if feat not in sample.columns:
            continue
        sub = sample.copy()
        arr = sub[feat].to_numpy()
        rng.shuffle(arr)
        sub[feat] = arr
        mse_perm = predict_and_score(sub)
        rows.append(
            {
                "feature": feat,
                "n_sample": n_sample,
                "mse_weighted_baseline": base_mse,
                "mse_weighted_permuted": mse_perm,
                "delta_mse_weighted": mse_perm - base_mse,
                "delta_mse_pct": (mse_perm - base_mse) / base_mse if base_mse else float("inf"),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_mse_weighted", ascending=False)


def add_predictions_and_residuals(
    df: pd.DataFrame,
    reg_main_map,
    sigma_main_per_regime: dict,
    feature_cols,
    regime_defs,
    use_return_scaling: bool,
    calibration_main_per_regime: dict,
) -> pd.DataFrame:
    """
    Given a dataframe with features and UNCLIPPED future_return, add:

      - regime (based on mid_price)
      - future_return_hat
      - net_return_hat (after tax)
      - prob_profit (Win%) via Normal CDF using sigma_main
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
        y_hat_scaled = reg.predict(X_reg)
        if use_return_scaling:
            scale_vec = df.loc[mask, "return_scale"].to_numpy(dtype="float64")
            y_hat = y_hat_scaled * scale_vec
        else:
            y_hat = y_hat_scaled
        calib = calibration_main_per_regime.get(regime_name, {"slope": 1.0, "intercept": 0.0})
        y_hat = float(calib.get("slope", 1.0)) * y_hat + float(calib.get("intercept", 0.0))
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
    df["net_return"] = (1.0 + df["future_return"]) * (1.0 - TAX_RATE) - 1.0
    df["profit_ok"] = (df["net_return"] > MARGIN).astype(int)
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
    df["price_bin"] = pd.cut(df["mid_price"], bins=price_edges, labels=price_labels, right=False, include_lowest=True)

    # Predicted Win% bins
    win_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
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

        y_true = sub["future_return"].to_numpy(dtype="float64")
        y_pred = sub["future_return_hat"].to_numpy(dtype="float64")
        y_net = sub["net_return"].to_numpy(dtype="float64")
        net_pred = sub["net_return_hat"].to_numpy(dtype="float64")
        mae = float(np.mean(np.abs(resid)))
        mae_w = weighted_mae(y_true, y_pred, sub["sample_weight"].to_numpy(dtype="float64"))
        r2 = float(1.0 - np.mean(resid ** 2) / np.var(y_true)) if np.var(y_true) > 0 else float("nan")
        r2_w = weighted_r2(y_true, y_pred, sub["sample_weight"].to_numpy(dtype="float64"))
        calib = calibration_stats(y_true, y_pred, sub["sample_weight"].to_numpy(dtype="float64"))

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
                "mean_net_return": float(np.mean(y_net)),
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "rmse_weighted": float(np.sqrt(mse)),
                "mae": mae,
                "mae_weighted": mae_w,
                "r2": r2,
                "r2_weighted": r2_w,
                "calibration_slope": calib["slope"],
                "calibration_intercept": calib["intercept"],
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
            "mean_net_return",
            "rmse",
            "rmse_weighted",
            "mae",
            "mae_weighted",
            "r2",
            "r2_weighted",
            "calibration_slope",
            "calibration_intercept",
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

    def write_and_push(out_df: pd.DataFrame, local_path: str, key_stub: str, reason: str = ""):
        """Write CSV locally (always) and push to R2 with a consistent stub."""
        out_df.to_csv(local_path, index=False)
        print(f"[save] Wrote {local_path} (rows={len(out_df)}){(' - ' + reason) if reason else ''}")

        now = datetime.now(timezone.utc)
        key = f"{prefix}/{key_stub}_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        buf = io.BytesIO(out_df.to_csv(index=False).encode("utf-8"))
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        print(f"[save] Wrote r2://{bucket}/{key}")

    empty_cols = [
        "time_bin", "price_bin", "win_bin", "pred_ret_bin",
        "n_rows", "sigma_unweighted", "sigma_weighted",
        "mean_future_return", "mean_pred_return",
        "mean_prob_profit", "mean_pred_net_return",
    ]

    df = build_eval_dataframe(s3, bucket)
    if df.empty:
        write_and_push(pd.DataFrame(columns=empty_cols), "sigma_buckets_full.csv", "sigma_buckets", reason="no eval data")
        # Still write an empty regime summary for completeness
        pd.DataFrame(columns=["regime", "n_rows", "sigma_unweighted", "sigma_weighted"]).to_csv(
            "sigma_regimes.csv", index=False
        )
        return

    # 2) Load regressors and meta
    (
        reg_main_map,
        sigma_main_per_regime,
        feature_cols,
        regime_defs,
        return_scaling_cfg,
        calibration_main_per_regime,
    ) = load_latest_regressor(s3, bucket)
    print(f"[main] Loaded main-horizon regressors for regimes: {list(reg_main_map.keys())}")
    print(f"[main] sigma_main_per_regime: {sigma_main_per_regime}")
    print(f"[main] Using {len(feature_cols)} features:", feature_cols)
    use_return_scaling = (return_scaling_cfg.get("type") == "volatility_tick")
    tick_size = float(return_scaling_cfg.get("tick_size", 1.0))
    min_scale = float(return_scaling_cfg.get("min_scale", 1e-3))
    max_scale = float(return_scaling_cfg.get("max_scale", 5.0))
    if use_return_scaling:
        df["return_scale"] = compute_return_scale(
            df,
            tick_size=tick_size,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    else:
        df["return_scale"] = 1.0

    # 3) Add predictions, residuals, Win%, regimes
    df = add_predictions_and_residuals(
        df,
        reg_main_map,
        sigma_main_per_regime,
        feature_cols,
        regime_defs,
        use_return_scaling,
        calibration_main_per_regime,
    )

    # 4) Add bins (time, price, Win%, predicted return)
    df = add_bins(df)

    # 4b) Quick diagnostics: scaling and volatility in the late-day high-return hotspot
    hotspot_mask = (df["time_bin"] == "20-24") & (df["pred_ret_bin"] == ">20%")
    hotspot = df.loc[hotspot_mask, ["return_scale", "volatility_60m"]]
    if not hotspot.empty:
        qs = [1, 5, 25, 50, 75, 95, 99]
        def fmt(series):
            return {f"p{q}": float(np.percentile(series, q)) for q in qs}
        print("[diag] Hotspot rows (time 20-24 & pred_ret_bin >20%):", len(hotspot))
        print("[diag] return_scale percentiles:", fmt(hotspot["return_scale"]))
        print("[diag] volatility_60m percentiles:", fmt(hotspot["volatility_60m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)))
    else:
        print("[diag] No rows in hotspot (time 20-24 & pred_ret_bin >20%).")

    # 4c) Global diagnostics
    overall = compute_overall_metrics(df)
    prob_stats = compute_prob_metrics(df)
    feature_corr = compute_feature_correlations(df, feature_cols)
    perm_importance = compute_permutation_importance(
        df,
        reg_main_map,
        sigma_main_per_regime,
        feature_cols,
        regime_defs,
        use_return_scaling,
        calibration_main_per_regime,
        sample_rows=PERMUTATION_SAMPLE_ROWS,
    )
    write_and_push(overall, "sigma_overall.csv", "sigma_overall")
    if not prob_stats.empty:
        write_and_push(prob_stats, "sigma_prob_metrics.csv", "sigma_prob")
    if not feature_corr.empty:
        write_and_push(feature_corr, "feature_correlations.csv", "feature_correlations")
    if not perm_importance.empty:
        write_and_push(perm_importance, "permutation_importance.csv", "permutation_importance")

    # 5) Compute bucket sigmas
    df_buckets = compute_bucket_sigmas(df)

    if df_buckets.empty:
        write_and_push(pd.DataFrame(columns=empty_cols), "sigma_buckets_full.csv", "sigma_buckets", reason="no buckets met min rows")
        df_reg = compute_regime_summaries(df)
        write_and_push(df_reg, "sigma_regimes.csv", "sigma_regimes")
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

    write_and_push(out_csv, "sigma_buckets_full.csv", "sigma_buckets")

    # Compute and save per-regime summary
    df_reg = compute_regime_summaries(df)
    write_and_push(df_reg, "sigma_regimes.csv", "sigma_regimes")
    print(f"[main] Wrote per-regime summary to sigma_regimes.csv:")
    print(df_reg.to_string(index=False))

    # Print marginal summaries (small enough to paste into chat)
    print_marginal_summaries(df_buckets)

if __name__ == "__main__":
    main()
