import argparse
import io
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import erf, sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from features import (
    add_model_features,
    add_multi_horizon_returns,
    compute_return_scale,
    flatten_5m_snapshots,
    get_r2_client,
    list_keys_with_prefix,
)

# Core economic params (aligned with training/scoring)
HORIZON_MINUTES = 60
TAX_RATE = 0.02
MARGIN = 0.002
# Recency decay for sample weights (matches training)
DECAY_DAYS = 14
# Regime clipping (same as training defaults)
REGIME_DEFS_DEFAULT = {
    "low": {"mid_price_min": 0, "mid_price_max": 10_000},
    "mid": {"mid_price_min": 10_000, "mid_price_max": 100_000},
    "high": {"mid_price_min": 100_000, "mid_price_max": None},
}
REGIME_CLIPPING = {
    "high": {"min": -0.40, "max": 0.40},
    "mid": {"min": -0.60, "max": 0.60},
    "low": {"min": -0.80, "max": 0.80},
}


def normal_cdf_array(z: np.ndarray) -> np.ndarray:
    def _cdf_scalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    vec = np.vectorize(_cdf_scalar, otypes=[float])
    return vec(z)


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
    """Load 5m snapshots inclusive of start/end dates (date objects)."""
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


def add_features_and_labels(df_raw, use_return_scaling: bool, tick_size: float, min_scale: float, max_scale: float, regime_defs):
    """
    Build features/targets. No predictions; used for both training and testing folds.
    """
    df = add_model_features(df_raw)
    df = add_multi_horizon_returns(df, horizons_minutes=[HORIZON_MINUTES])
    col_target = f"ret_{HORIZON_MINUTES}m"
    df = df.dropna(subset=[col_target]).copy()
    df["future_return"] = df[col_target]

    df["regime"] = df["mid_price"].apply(lambda p: assign_regime_for_price(p, regime_defs))

    if use_return_scaling:
        df["return_scale"] = compute_return_scale(
            df,
            tick_size=tick_size,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    else:
        df["return_scale"] = 1.0

    df["net_return"] = (1.0 + df["future_return"]) * (1.0 - TAX_RATE) - 1.0
    df["profit_ok"] = (df["net_return"] > MARGIN).astype(int)
    return df


def clip_returns_for_regime(y: np.ndarray, regime_name: str) -> np.ndarray:
    cfg = REGIME_CLIPPING.get(regime_name)
    if cfg is None:
        return y
    return np.clip(y, cfg["min"], cfg["max"])


def compute_sample_weight(ts: pd.Series) -> np.ndarray:
    ref = ts.max()
    age_days = (ref - ts).dt.total_seconds() / (3600 * 24)
    return np.exp(-age_days / DECAY_DAYS)


def apply_policy_filters(df, policy):
    mask = np.ones(len(df), dtype=bool)
    if policy.get("prob_min") is not None:
        mask &= df["prob_profit"] >= policy["prob_min"]
    if policy.get("ret_min") is not None:
        mask &= df["net_return_hat"] >= policy["ret_min"]
    if policy.get("block_low_after_20"):
        ts = df["timestamp"]
        hours = ts.dt.hour + ts.dt.minute / 60.0
        mask &= ~((df["regime"] == "low") & (hours >= 20))
    if policy.get("cap_net_return_low") is not None:
        low_mask = df["regime"] == "low"
        df.loc[low_mask, "net_return_hat"] = np.minimum(
            df.loc[low_mask, "net_return_hat"].values,
            policy["cap_net_return_low"],
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
    return 0.002


def simulate_trades(df, policy, slippage):
    trades_mask = apply_policy_filters(df.copy(), policy)
    trades = df.loc[trades_mask].copy()
    if trades.empty:
        return trades
    slip = trades.apply(lambda r: slippage_per_side(r, slippage), axis=1).to_numpy(dtype="float64")
    net_realised = trades["net_return"].to_numpy(dtype="float64") - 2.0 * slip
    trades["net_realised"] = net_realised
    trades["slippage_per_side"] = slip
    return trades


def summarize_trades(trades: pd.DataFrame, policy_name: str, slippage: str, fold_id: str):
    if trades.empty:
        return {
            "fold_id": fold_id,
            "policy": policy_name,
            "slippage": slippage,
            "n_trades": 0,
            "hit_rate": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "sum": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }
    returns = trades["net_realised"].to_numpy(dtype="float64")
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(dd.max()) if len(dd) else 0.0
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9))
    return {
        "fold_id": fold_id,
        "policy": policy_name,
        "slippage": slippage,
        "n_trades": int(len(trades)),
        "hit_rate": float((returns > 0).mean()),
        "mean": float(np.mean(returns)),
        "median": float(np.median(returns)),
        "p95": float(np.percentile(returns, 95)),
        "p99": float(np.percentile(returns, 99)),
        "sum": float(np.sum(returns)),
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def train_fold_models(df_train: pd.DataFrame, feature_cols, regime_defs: dict) -> Tuple[dict, dict]:
    """
    Train regime-specific regressors for the fold and return sigma per regime.
    """
    reg_models = {}
    sigma_per_regime = {}

    regimes = list(regime_defs.keys()) if regime_defs else list(df_train["regime"].unique())
    for regime_name in regimes:
        sub = df_train[df_train["regime"] == regime_name].copy()
        if sub.empty:
            continue
        y = sub["future_return"].to_numpy(dtype="float64")
        y = clip_returns_for_regime(y, regime_name)
        X = sub[feature_cols].to_numpy(dtype="float64")
        scale = sub["return_scale"].to_numpy(dtype="float64")
        scale = np.where((np.isfinite(scale)) & (scale > 0), scale, 1.0)
        y_scaled = y / scale

        w = sub.get("sample_weight")
        if w is None:
            w = np.ones(len(sub), dtype="float64")
        else:
            w = w.to_numpy(dtype="float64")
            if not np.any(np.isfinite(w)):
                w = np.ones(len(sub), dtype="float64")

        reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=4,
            tree_method="hist",
        )
        reg.fit(X, y_scaled, sample_weight=w)
        reg_models[regime_name] = reg

        y_pred = reg.predict(X) * scale
        resid = y - y_pred
        # weighted residual std
        if not np.any(np.isfinite(w)):
            w = np.ones_like(resid)
        w = w / np.mean(w)
        sigma = float(np.sqrt(np.average(resid ** 2, weights=w)))
        sigma_per_regime[regime_name] = max(sigma, 1e-8)

    return reg_models, sigma_per_regime


def predict_with_models(df: pd.DataFrame, reg_models: dict, feature_cols, sigma_per_regime: dict) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)
    future_return_hat = np.full(n_rows, np.nan, dtype="float64")
    for regime_name, reg in reg_models.items():
        mask = df["regime"] == regime_name
        if not mask.any():
            continue
        X_reg = df.loc[mask, feature_cols].to_numpy(dtype="float64")
        preds_scaled = reg.predict(X_reg)
        preds = preds_scaled * df.loc[mask, "return_scale"].to_numpy(dtype="float64")
        future_return_hat[mask.values] = preds

    df["future_return_hat"] = future_return_hat
    df["net_return_hat"] = (1.0 + df["future_return_hat"]) * (1.0 - TAX_RATE) - 1.0

    sigma_vals = np.array(list(sigma_per_regime.values()), dtype="float64")
    valid_sigma = sigma_vals[np.isfinite(sigma_vals) & (sigma_vals > 0)]
    sigma_fallback = float(np.median(valid_sigma)) if valid_sigma.size > 0 else 0.02
    df["regime_sigma"] = df["regime"].map(sigma_per_regime).astype(float)
    df["regime_sigma"] = df["regime_sigma"].fillna(sigma_fallback)

    gross_threshold = (1.0 + MARGIN) / (1.0 - TAX_RATE) - 1.0
    z = (df["future_return_hat"].values - gross_threshold) / np.maximum(df["regime_sigma"].values, 1e-8)
    df["prob_profit"] = np.clip(normal_cdf_array(z), 1e-4, 1.0 - 1e-4)
    return df


@dataclass
class FoldWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    fold_id: str


def build_folds(start_date: datetime, end_date: datetime, train_days: int, test_days: int, embargo_minutes: int, anchored: bool) -> List[FoldWindow]:
    folds: List[FoldWindow] = []
    train_len = timedelta(days=train_days)
    test_len = timedelta(days=test_days)
    embargo = timedelta(minutes=embargo_minutes)

    train_start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    train_end = train_start + train_len
    end_limit = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)

    while True:
        test_start = train_end + embargo
        test_end = test_start + test_len
        if test_start >= end_limit:
            break
        fold_id = f"{test_start.date()}_{test_end.date()}"
        folds.append(FoldWindow(train_start, train_end, test_start, test_end, fold_id))
        if anchored:
            # keep train_start fixed, extend train_end forward
            train_end = train_end + test_len
        else:
            train_start = train_start + test_len
            train_end = train_start + train_len
    return folds


def run_backtest(params):
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    meta_key = params.meta_key or "models/quantile/latest_meta.json"
    # Load meta for feature list and return-scaling config (models are trained per fold)
    obj_meta = s3.get_object(Bucket=bucket, Key=meta_key)
    meta = pd.read_json(io.BytesIO(obj_meta["Body"].read()), typ="series").to_dict()

    feature_cols = meta.get(
        "feature_cols",
        ["mid_price", "spread_pct", "log_volume_5m"],
    )
    regime_defs = meta.get("regime_defs") or REGIME_DEFS_DEFAULT
    return_scaling_cfg = meta.get("return_scaling") or {}
    use_return_scaling = return_scaling_cfg.get("type") == "volatility_tick"
    tick_size = float(return_scaling_cfg.get("tick_size", 1.0))
    min_scale = float(return_scaling_cfg.get("min_scale", 1e-3))
    max_scale = float(return_scaling_cfg.get("max_scale", 5.0))

    df_raw = load_snapshots_between(s3, bucket, params.start_date.date(), params.end_date.date())
    if df_raw.empty:
        raise RuntimeError("No snapshots found for given date range.")

    df_all = add_features_and_labels(
        df_raw,
        use_return_scaling=use_return_scaling,
        tick_size=tick_size,
        min_scale=min_scale,
        max_scale=max_scale,
        regime_defs=regime_defs,
    )
    df_all = df_all[df_all["mid_price"] > 0].copy()

    folds = build_folds(
        params.start_date.date(),
        params.end_date.date(),
        params.train_window_days,
        params.test_window_days,
        params.embargo_minutes,
        params.anchored,
    )
    if not folds:
        raise RuntimeError("No folds generated; check date range and window lengths.")

    policies = [
        {"name": "baseline", "prob_min": None, "ret_min": 0.0},
        {"name": "prob_55", "prob_min": 0.55, "ret_min": 0.0},
        {"name": "prob_60", "prob_min": 0.60, "ret_min": 0.0},
        {"name": "prob_60_cap_low", "prob_min": 0.60, "ret_min": 0.0, "cap_net_return_low": 0.25},
    ]
    slippages = ["spread_plus_tick", "tiered", "regime_based"]

    summaries = []
    trades_all = []

    for fold in folds:
        # Train subset with embargo: ensure labels do not reach into test window
        train_cutoff = fold.test_start - timedelta(minutes=HORIZON_MINUTES)
        train_mask = (df_all["timestamp"] >= fold.train_start) & (df_all["timestamp"] < fold.train_end)
        train_mask &= df_all["timestamp"] < train_cutoff
        df_train = df_all.loc[train_mask].copy()
        if df_train.empty:
            continue

        df_train["sample_weight"] = compute_sample_weight(df_train["timestamp"])

        reg_models, sigma_per_regime = train_fold_models(df_train, feature_cols, regime_defs)
        if not reg_models:
            continue

        df_test = df_all[
            (df_all["timestamp"] >= fold.test_start) & (df_all["timestamp"] < fold.test_end)
        ].copy()
        if df_test.empty:
            continue

        df_test = predict_with_models(df_test, reg_models, feature_cols, sigma_per_regime)

        for pol in policies:
            for slip in slippages:
                trades = simulate_trades(df_test, pol, slip)
                if not trades.empty:
                    trades["fold_id"] = fold.fold_id
                    trades["policy"] = pol["name"]
                    trades["slippage"] = slip
                    trades_all.append(trades)
                summary = summarize_trades(trades, pol["name"], slip, fold.fold_id)
                summaries.append(summary)

    if trades_all:
        trades_df = pd.concat(trades_all, ignore_index=True)
    else:
        trades_df = pd.DataFrame()

    summaries_df = pd.DataFrame(summaries)
    agg = (
        summaries_df.groupby(["policy", "slippage"])
        .agg(
            n_folds=("fold_id", "nunique"),
            n_trades=("n_trades", "sum"),
            hit_rate=("hit_rate", "mean"),
            mean_return=("mean", "mean"),
            median_return=("median", "mean"),
            p95_return=("p95", "mean"),
            sum_return=("sum", "sum"),
            sharpe_mean=("sharpe", "mean"),
            max_dd_mean=("max_dd", "mean"),
        )
        .reset_index()
    )

    trades_df.to_csv("backtest_v2_trades.csv", index=False)
    summaries_df.to_csv("backtest_v2_folds.csv", index=False)
    agg.to_csv("backtest_v2_summary.csv", index=False)

    print("Fold-level summary:")
    print(summaries_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nAggregate by policy/slippage:")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nWrote backtest_v2_trades.csv, backtest_v2_folds.csv, backtest_v2_summary.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward backtest with embargoed folds.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--meta-key", default=None, help="R2 key for meta JSON (for feature list/return scaling)")
    parser.add_argument("--train-window-days", type=int, default=4, help="Days in each train window for per-fold training")
    parser.add_argument("--test-window-days", type=int, default=1, help="Days per fold test window")
    parser.add_argument("--embargo-minutes", type=int, default=60, help="Minutes to embargo between train/test windows (protect label leakage)")
    parser.add_argument("--anchored", action="store_true", help="Use anchored expanding train windows instead of rolling")
    args = parser.parse_args()
    args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return args


if __name__ == "__main__":
    run_backtest(parse_args())
