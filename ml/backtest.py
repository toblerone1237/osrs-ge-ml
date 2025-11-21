import argparse
import io
import os
from datetime import datetime, timedelta, timezone

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

# Economic params (aligned with training/scoring)
HORIZON_MINUTES = 60
TAX_RATE = 0.02
MARGIN = 0.002


def load_model_and_meta(s3, bucket, reg_key="models/quantile/latest_reg.pkl", meta_key="models/quantile/latest_meta.json"):
    obj_reg = s3.get_object(Bucket=bucket, Key=reg_key)
    obj_meta = s3.get_object(Bucket=bucket, Key=meta_key)
    reg_models = joblib.load(io.BytesIO(obj_reg["Body"].read()))
    meta = pd.read_json(io.BytesIO(obj_meta["Body"].read()), typ="series").to_dict()
    return reg_models, meta


def assign_regime_for_price(mid_price: float, regime_defs) -> str:
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


def load_snapshots_between(s3, bucket, start_date, end_date):
    """Inclusive of start_date/end_date (date objects)."""
    chunks = []
    d = start_date
    while d <= end_date:
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if keys:
            df_day = flatten_5m_snapshots(s3, bucket, keys)
            if not df_day.empty:
                chunks.append(df_day)
        d += timedelta(days=1)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df.sort_values(["item_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_labels_and_predictions(df_raw, reg_models, meta):
    """Add features, target (future_return), predicted returns, bins, etc."""
    df = add_model_features(df_raw)
    df = add_multi_horizon_returns(df, horizons_minutes=[HORIZON_MINUTES])
    col_target = f"ret_{HORIZON_MINUTES}m"
    df = df.dropna(subset=[col_target]).copy()
    df["future_return"] = df[col_target]

    # Config
    feature_cols = meta.get("feature_cols", ["mid_price", "spread_pct", "log_volume_5m"])
    regime_defs = meta.get("regime_defs") or {
        "low": {"mid_price_min": 0, "mid_price_max": 10_000},
        "mid": {"mid_price_min": 10_000, "mid_price_max": 100_000},
        "high": {"mid_price_min": 100_000, "mid_price_max": None},
    }
    return_scaling_cfg = meta.get("return_scaling") or {}
    use_return_scaling = return_scaling_cfg.get("type") == "volatility_tick"
    tick_size = float(return_scaling_cfg.get("tick_size", 1.0))
    min_scale = float(return_scaling_cfg.get("min_scale", 1e-3))
    max_scale = float(return_scaling_cfg.get("max_scale", 5.0))
    sigma_main_per_regime = meta.get("sigma_main_per_regime") or {}
    calibration_main_per_regime = meta.get("calibration_main_per_regime") or {}

    # Assign regimes
    df["regime"] = df["mid_price"].apply(lambda p: assign_regime_for_price(p, regime_defs))

    # Scaling factor
    if use_return_scaling:
        df["return_scale"] = compute_return_scale(
            df,
            tick_size=tick_size,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    else:
        df["return_scale"] = 1.0

    # Predict main horizon per regime
    n_rows = len(df)
    future_return_hat = np.full(n_rows, np.nan, dtype="float64")
    for regime_name, models_for_regime in reg_models.items():
        reg_h = models_for_regime.get(HORIZON_MINUTES)
        if reg_h is None:
            continue
        mask = df["regime"] == regime_name
        if not mask.any():
            continue
        X_reg = df.loc[mask, feature_cols].values
        preds_scaled = reg_h.predict(X_reg)
        preds = preds_scaled * df.loc[mask, "return_scale"].to_numpy(dtype="float64")
        calib = calibration_main_per_regime.get(regime_name, {"slope": 1.0, "intercept": 0.0})
        preds = float(calib.get("slope", 1.0)) * preds + float(calib.get("intercept", 0.0))
        future_return_hat[mask.values] = preds

    df["future_return_hat"] = future_return_hat
    df["net_return_hat"] = (1.0 + df["future_return_hat"]) * (1.0 - TAX_RATE) - 1.0

    # Regime sigma map (fallback to global median)
    sigma_vals = np.array(list(sigma_main_per_regime.values()), dtype="float64")
    valid_sigma = sigma_vals[np.isfinite(sigma_vals) & (sigma_vals > 0)]
    if valid_sigma.size > 0:
        sigma_fallback = float(np.median(valid_sigma))
    else:
        sigma_fallback = float(meta.get("sigma_main", 0.02))
    df["regime_sigma"] = df["regime"].map(sigma_main_per_regime).astype(float)
    df["regime_sigma"].fillna(sigma_fallback, inplace=True)

    gross_threshold = (1.0 + MARGIN) / (1.0 - TAX_RATE) - 1.0
    z = (df["future_return_hat"].values - gross_threshold) / np.maximum(df["regime_sigma"].values, 1e-8)
    df["prob_profit"] = np.clip(0.5 * (1.0 + np.erf(z / np.sqrt(2.0))), 1e-4, 1.0 - 1e-4)

    # Bins for filtering
    ts = df["timestamp"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(timezone.utc)
    else:
        ts = ts.dt.tz_localize(timezone.utc)
    hours = ts.dt.hour + ts.dt.minute / 60.0
    df["time_bin"] = pd.cut(hours, bins=[0, 4, 8, 12, 16, 20, 24], labels=["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"], right=False, include_lowest=True)
    df["price_bin"] = pd.cut(
        df["mid_price"],
        bins=[-np.inf, 1_000, 10_000, 100_000, 1_000_000, np.inf],
        labels=["<=1k", "1k-10k", "10k-100k", "100k-1m", ">1m"],
        right=False,
        include_lowest=True,
    )
    df["win_bin"] = pd.cut(
        df["prob_profit"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        right=False,
        include_lowest=True,
    )
    df["pred_ret_bin"] = pd.cut(
        df["net_return_hat"],
        bins=[-1.0, -0.05, 0.0, 0.05, 0.2, np.inf],
        labels=["<-5%", "-5-0%", "0-5%", "5-20%", ">20%"],
        right=False,
        include_lowest=True,
    )

    return df


def apply_policy_filters(df, policy):
    """Return mask of rows to trade."""
    mask = np.ones(len(df), dtype=bool)
    if policy.get("prob_min") is not None:
        mask &= df["prob_profit"] >= policy["prob_min"]
    if policy.get("ret_min") is not None:
        mask &= df["net_return_hat"] >= policy["ret_min"]
    if policy.get("block_low_after_20"):
        mask &= ~((df["regime"] == "low") & (df["time_bin"] == "20-24"))
    if policy.get("block_hotspot"):
        mask &= ~(
            (df["pred_ret_bin"] == ">20%")
            & (df["win_bin"] == "80-100%")
            & (df["time_bin"] == "20-24")
        )
    if policy.get("block_smallcap_20_24"):
        mask &= ~((df["price_bin"] == "<=1k") & (df["time_bin"] == "20-24"))
    cap = policy.get("cap_net_return_low")
    if cap is not None:
        low_mask = df["regime"] == "low"
        df.loc[low_mask, "net_return_hat"] = np.minimum(
            df.loc[low_mask, "net_return_hat"].values,
            cap,
        )
    return mask


def slippage_per_side(row, preset):
    spread = float(row.get("spread_pct", 0.0))
    price = float(row.get("mid_price", 0.0))
    regime = row.get("regime")
    if preset == "spread_plus_tick":
        return max(spread, 0.001)
    if preset == "tiered":
        if price >= 100_000:
            return 0.002
        if price >= 10_000:
            return 0.003
        return 0.008
    if preset == "regime_based":
        if regime == "high":
            return 0.002
        if regime == "mid":
            return 0.004
        return 0.008
    return 0.002  # default fallback


def simulate(df, policy, slippage_preset):
    trades_mask = apply_policy_filters(df.copy(), policy)
    trades = df.loc[trades_mask].copy()
    if trades.empty:
        return {"n_trades": 0}

    # Realised net return after tax + slippage both sides
    slip = trades.apply(lambda r: slippage_per_side(r, slippage_preset), axis=1).to_numpy()
    net_realised = (1.0 + trades["future_return"].to_numpy()) * (1.0 - TAX_RATE) - 1.0
    net_realised -= 2.0 * slip  # pay slippage on entry+exit

    trades["net_realised"] = net_realised

    return {
        "n_trades": int(len(trades)),
        "hit_rate": float((net_realised > 0).mean()) if len(trades) else 0.0,
        "mean": float(np.mean(net_realised)) if len(trades) else 0.0,
        "median": float(np.median(net_realised)) if len(trades) else 0.0,
        "p95": float(np.percentile(net_realised, 95)) if len(trades) else 0.0,
        "p99": float(np.percentile(net_realised, 99)) if len(trades) else 0.0,
        "sum": float(np.sum(net_realised)),
    }


def run_backtest(start_date, end_date, reg_key=None, meta_key=None):
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    reg_key = reg_key or "models/quantile/latest_reg.pkl"
    meta_key = meta_key or "models/quantile/latest_meta.json"

    df_raw = load_snapshots_between(s3, bucket, start_date, end_date)
    if df_raw.empty:
        raise RuntimeError("No snapshots found for given date range.")

    reg_models, meta = load_model_and_meta(s3, bucket, reg_key, meta_key)
    df = add_labels_and_predictions(df_raw, reg_models, meta)

    policies = [
        {"name": "baseline", "prob_min": None, "ret_min": 0.0},
        {"name": "prob_55", "prob_min": 0.55, "ret_min": 0.0},
        {"name": "prob_60", "prob_min": 0.60, "ret_min": 0.0},
        {"name": "prob_60_cap_low", "prob_min": 0.60, "ret_min": 0.0, "cap_net_return_low": 0.25},
        {"name": "prob_60_block_low20", "prob_min": 0.60, "ret_min": 0.0, "block_low_after_20": True},
        {"name": "prob_60_block_hotspot", "prob_min": 0.60, "ret_min": 0.0, "block_hotspot": True},
        {"name": "prob_60_block_smallcap_20_24", "prob_min": 0.60, "ret_min": 0.0, "block_smallcap_20_24": True},
    ]
    slippages = ["spread_plus_tick", "tiered", "regime_based"]

    results = []
    for pol in policies:
        for slip in slippages:
            stats = simulate(df, pol, slip)
            stats["policy"] = pol["name"]
            stats["slippage"] = slip
            results.append(stats)

    res_df = pd.DataFrame(results)
    res_df = res_df[
        ["policy", "slippage", "n_trades", "hit_rate", "mean", "median", "p95", "p99", "sum"]
    ].sort_values(["policy", "slippage"])
    print("\n=== Backtest results ===")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    res_df.to_csv("backtest_results.csv", index=False)
    print("Wrote backtest_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple backtest over a date range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--reg-key", default=None, help="S3 key for regressor pickle")
    parser.add_argument("--meta-key", default=None, help="S3 key for meta JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    run_backtest(start_date, end_date, reg_key=args.reg_key, meta_key=args.meta_key)
