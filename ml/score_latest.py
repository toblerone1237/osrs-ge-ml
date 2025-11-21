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
    compute_return_scale,
    HORIZONS_MINUTES,  # [5, 10, ..., 120]
)

EXPERIMENT = "quantile"

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

HORIZON_MINUTES = 60  # main decision horizon (must be in HORIZONS_MINUTES)
WINDOW_MINUTES = 60   # how far back to pull snapshots for scoring
TAX_RATE = 0.02       # same as in training
MARGIN = 0.002        # same as in training

# Minimum volume over the scoring window for an item to be considered "active"
MIN_VOLUME_WINDOW = 1

# sanity only now; we don't use this as a cutoff in the main pipeline anymore
MIN_MID_PRICE = 0

# Default regimes (used if meta doesn't define any)
REGIME_DEFS_DEFAULT = {
    "low":  {"mid_price_min": 0,       "mid_price_max": 10_000},
    "mid":  {"mid_price_min": 10_000,  "mid_price_max": 100_000},
    "high": {"mid_price_min": 100_000, "mid_price_max": None},
}


# ---------------------------------------------------------------------------
# Helpers
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


def get_latest_5m_key(s3, bucket):
    """
    Return the key of the most recent 5m snapshot in the bucket, or None.
    """
    now = datetime.now(timezone.utc)
    prefix = f"5m/{now.year}/{now.month:02d}/{now.day:02d}/"
    keys = list_keys_with_prefix(s3, bucket, prefix)
    if not keys:
        return None

    keys.sort()
    return keys[-1]


def select_recent_keys_by_name(all_keys, latest_key, window_minutes, bucket_minutes=5):
    """
    Pick the most recent keys that should cover the window, based on filename order.
    Assumes keys are named 5m/YYYY/MM/DD/HH-MM.json so lexical order == time order.
    """
    if not all_keys:
        return []

    all_keys = sorted(all_keys)
    try:
        latest_idx = all_keys.index(latest_key)
    except ValueError:
        latest_idx = len(all_keys) - 1

    # Add a small buffer (+1) to account for alignment/fuzz in bucket timing
    buckets_needed = max(1, int(np.ceil(window_minutes / bucket_minutes)) + 1)
    start_idx = max(0, latest_idx - buckets_needed + 1)
    return all_keys[start_idx : latest_idx + 1]


def load_latest_models(s3, bucket):
    """
    Load the latest regressors and metadata from R2.
    """
    key_reg = "models/quantile/latest_reg.pkl"
    key_meta = "models/quantile/latest_meta.json"

    obj_reg = s3.get_object(Bucket=bucket, Key=key_reg)
    obj_meta = s3.get_object(Bucket=bucket, Key=key_meta)

    reg_models = joblib.load(io.BytesIO(obj_reg["Body"].read()))
    meta = pd.read_json(io.BytesIO(obj_meta["Body"].read()), typ="series").to_dict()

    return reg_models, meta


def build_scoring_dataframe(s3, bucket):
    """
    Load recent 5m snapshots into a DataFrame, restricted to WINDOW_MINUTES.
    """
    latest_key = get_latest_5m_key(s3, bucket)
    if not latest_key:
        raise RuntimeError("No latest 5m snapshot found in bucket.")

    # All snapshots for that day
    day_prefix = latest_key.rsplit("/", 1)[0] + "/"
    all_keys = list_keys_with_prefix(s3, bucket, day_prefix)

    # Determine the latest timestamp (fetch only the latest file once)
    obj_latest = s3.get_object(Bucket=bucket, Key=latest_key)
    import json as _json
    latest_json = _json.loads(obj_latest["Body"].read())
    latest_ts = latest_json["five_minute"]["timestamp"]

    # Choose last-hour keys purely by filename ordering (no per-file downloads)
    recent_keys = select_recent_keys_by_name(
        all_keys, latest_key, WINDOW_MINUTES, bucket_minutes=5
    )

    print(
        f"Using {len(recent_keys)} snapshots from the last {WINDOW_MINUTES} minutes (by filename ordering)."
    )

    df = flatten_5m_snapshots(s3, bucket, recent_keys)
    if df.empty:
        raise RuntimeError("No rows in scoring DataFrame after flatten.")

    df = add_model_features(df)
    df = df[df["mid_price"] > 0].copy()  # only sanity, no hard cutoff
    return df, latest_ts


def compute_window_aggregates(df, window_start_ts):
    """
    Compute aggregated statistics over the scoring window per item.
    """
    df = df.copy()
    df = df[df["timestamp_unix"] >= window_start_ts]

    grouped = df.groupby("item_id").agg(
        {
            "total_volume_5m": "sum",
            "mid_price": ["first", "last"],
        }
    )
    grouped.columns = ["volume_window", "price_start", "price_end"]
    grouped.reset_index(inplace=True)

    grouped["window_return"] = grouped["price_end"] / grouped["price_start"] - 1.0
    return grouped


def filter_active_items(df_agg):
    """
    Filter items that meet minimum volume criteria.
    """
    df = df_agg.copy()
    df = df[df["volume_window"] >= MIN_VOLUME_WINDOW].copy()
    # no mid-price cutoff anymore
    return df


def normal_cdf_array(z: np.ndarray) -> np.ndarray:
    """
    Vectorised Normal(0,1) CDF using math.erf.
    """
    def _cdf_scalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    vec = np.vectorize(_cdf_scalar, otypes=[float])
    return vec(z)


def load_latest_item_mapping(s3, bucket):
    """
    Load the latest daily snapshot and derive item id->name mapping.
    Reuses the logic from osrs-ge-daily (extract_item_mapping_from_daily).
    """
    # We'll scan back up to 7 days for a daily snapshot
    now = datetime.now(timezone.utc)
    for delta in range(7):
        d = now - timedelta(days=delta)
        prefix = f"daily/{d.year}/{d.month:02d}/{d.day:02d}"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if not keys:
            continue
        keys.sort()
        latest_key = keys[-1]
        obj = s3.get_object(Bucket=bucket, Key=latest_key)
        buf = io.BytesIO(obj["Body"].read())
        try:
            df_daily = pd.read_parquet(buf)
        except Exception:
            # Fallback: maybe it's JSON
            import json
            df_daily = pd.json_normalize(json.loads(buf.getvalue()))
        df_daily.attrs["source_key"] = latest_key
        return extract_item_mapping_from_daily(df_daily)

    # If nothing found, return empty mapping
    return pd.DataFrame(columns=["id", "name"])


def extract_item_mapping_from_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Given a daily snapshot dataframe, extract item id/name mapping.
    We try a few common shapes.
    """
    # If the daily snapshot already has "id" and "name" columns, use them
    if {"id", "name"}.issubset(df_daily.columns):
        return df_daily[["id", "name"]].drop_duplicates().reset_index(drop=True)

    # Otherwise, try to find nested mapping
    # (this mirrors the logic in osrs-ge-daily worker)
    mapping_rows = []

    def dfs(obj):
        if isinstance(obj, dict):
            if "id" in obj and "name" in obj:
                mapping_rows.append({"id": obj["id"], "name": obj["name"]})
            for v in obj.values():
                dfs(v)
        elif isinstance(obj, list):
            for v in obj:
                dfs(v)

    for _, row in df_daily.iterrows():
        dfs(row.to_dict())

    if not mapping_rows:
        return pd.DataFrame(columns=["id", "name"])

    mapping = pd.DataFrame(mapping_rows).drop_duplicates()
    return mapping.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    reg_models_raw, meta = load_latest_models(s3, bucket)

    # Meta / config
    horizon_minutes = int(meta.get("horizon_minutes", HORIZON_MINUTES))
    path_horizons = meta.get("path_horizons_minutes", HORIZONS_MINUTES)
    feature_cols = meta.get("feature_cols", ["mid_price", "spread_pct", "log_volume_5m"])
    tax_rate = float(meta.get("tax_rate", TAX_RATE))
    margin = float(meta.get("margin", MARGIN))
    sigma_main_global = float(meta.get("sigma_main", 0.02))

    sigma_main_per_regime_meta = meta.get("sigma_main_per_regime")
    regime_defs_meta = meta.get("regime_defs")
    return_scaling_cfg = meta.get("return_scaling") or {}
    use_return_scaling = return_scaling_cfg.get("type") == "volatility_tick"
    tick_size = float(return_scaling_cfg.get("tick_size", 1.0))
    min_return_scale = float(return_scaling_cfg.get("min_scale", 1e-4))
    max_return_scale = float(return_scaling_cfg.get("max_scale", 5.0))

    # Normalise model structure to: regime -> horizon -> regressor
    if not reg_models_raw:
        raise RuntimeError("reg_models is empty; nothing to score with.")

    first_key = next(iter(reg_models_raw.keys()))
    if isinstance(first_key, int):
        # Old format: single global dict[horizon] -> regressor
        reg_models = {"global": reg_models_raw}
        regime_defs = regime_defs_meta or {"global": {"mid_price_min": 0, "mid_price_max": None}}
        if isinstance(sigma_main_per_regime_meta, dict) and sigma_main_per_regime_meta:
            sigma_main_per_regime = {k: float(v) for k, v in sigma_main_per_regime_meta.items()}
        else:
            sigma_main_per_regime = {"global": sigma_main_global}
    else:
        # New format: regime_name -> {horizon: regressor}
        reg_models = reg_models_raw
        regime_defs = regime_defs_meta or REGIME_DEFS_DEFAULT
        if isinstance(sigma_main_per_regime_meta, dict) and sigma_main_per_regime_meta:
            sigma_main_per_regime = {k: float(v) for k, v in sigma_main_per_regime_meta.items()}
        else:
            # Fallback: assign global sigma to each regime
            sigma_main_per_regime = {
                regime_name: sigma_main_global for regime_name in reg_models.keys()
            }

    df_raw, latest_ts = build_scoring_dataframe(s3, bucket)
    window_start_ts = latest_ts - WINDOW_MINUTES * 60

    # Aggregate activity over the scoring window
    df_agg = compute_window_aggregates(df_raw, window_start_ts)
    df_active = filter_active_items(df_agg)
    if df_active.empty:
        print("No active items found in scoring window.")
        return

    df = df_raw.merge(df_active[["item_id"]], on="item_id", how="inner")

    if use_return_scaling:
        df["return_scale"] = compute_return_scale(
            df,
            tick_size=tick_size,
            min_scale=min_return_scale,
            max_scale=max_return_scale,
        )
    else:
        df["return_scale"] = 1.0

    # Assign regimes based on current mid_price
    df["regime"] = df["mid_price"].apply(
        lambda p: assign_regime_for_price(p, regime_defs)
    )

    print(
        f"Scoring {df['item_id'].nunique()} items with recent activity "
        f"across regimes: {df['regime'].value_counts().to_dict()}"
    )

    # Ensure stable index for array-based operations
    df = df.reset_index(drop=True)

    X_all = df[feature_cols].values
    n_rows = len(df)

    # For each path horizon, compute predictions regime-wise
    path_results = {}
    for H in path_horizons:
        preds_all = np.full(n_rows, np.nan, dtype="float64")
        for regime_name, models_for_regime in reg_models.items():
            reg_h = models_for_regime.get(H)
            if reg_h is None:
                continue
            mask = df["regime"] == regime_name
            if not mask.any():
                continue
            X_reg = df.loc[mask, feature_cols].values
            preds = reg_h.predict(X_reg)
            if use_return_scaling:
                scale_vec = df.loc[mask, "return_scale"].to_numpy(dtype="float64")
                preds = preds * scale_vec
            preds_all[mask.values] = preds
        path_results[H] = preds_all
        print(f"[score] Predicting horizon {H}m for {n_rows} rows (all regimes).")

    # Main horizon predictions
    if horizon_minutes not in path_results:
        raise RuntimeError(f"Missing regressor for main horizon {horizon_minutes}m.")
    df["future_return_hat"] = path_results[horizon_minutes]

    # Net return and expected profit
    df["net_return_hat"] = (1.0 + df["future_return_hat"]) * (1.0 - tax_rate) - 1.0
    df["buy_price"] = df["mid_price"]
    df["sell_price_hat"] = df["buy_price"] * (1.0 + df["net_return_hat"])
    df["expected_profit"] = df["sell_price_hat"] - df["buy_price"]

    hold_seconds = horizon_minutes * 60.0
    df["expected_profit_per_second"] = df["expected_profit"] / hold_seconds

    # Probability-of-profit from regime-specific Normal approximation
    gross_threshold = (1.0 + margin) / (1.0 - tax_rate) - 1.0

    df["regime_sigma"] = df["regime"].map(sigma_main_per_regime).astype(float)
    sigma_vec = df["regime_sigma"].values
    # Fallback to global sigma if missing or non-positive
    sigma_vec = np.where(
        (np.isfinite(sigma_vec)) & (sigma_vec > 0.0),
        sigma_vec,
        sigma_main_global,
    )
    z = (df["future_return_hat"].values - gross_threshold) / np.maximum(sigma_vec, 1e-8)
    prob_profit = normal_cdf_array(z)
    df["prob_profit"] = np.clip(prob_profit, 1e-4, 1.0 - 1e-4)

    # Summarise per item (last snapshot)
    df.sort_values(["item_id", "timestamp"], inplace=True)
    last_per_item = df.groupby("item_id").tail(1).copy()
    last_per_item = last_per_item.merge(df_agg, on="item_id", how="left")

    # Build path predictions for each item
    path_rows = []
    for _, r in last_per_item.iterrows():
        item_id = r["item_id"]
        row_path = []
        # Find the last index for this item in df
        mask = (df["item_id"] == item_id)
        row_idx = df.index[mask][-1]
        for h in sorted(path_horizons):
            preds_h = path_results.get(h)
            if preds_h is None:
                continue
            ret_hat = preds_h[row_idx]
            row_path.append({"minutes": int(h), "future_return_hat": float(ret_hat)})
        path_rows.append(row_path)

    last_per_item["path"] = path_rows

    # Compute a regime penalty based on sigma_main_per_regime
    sigma_vals = np.array(list(sigma_main_per_regime.values()), dtype="float64")
    sigma_vals = sigma_vals[np.isfinite(sigma_vals) & (sigma_vals > 0)]
    if sigma_vals.size > 0:
        if "mid" in sigma_main_per_regime:
            base_sigma = sigma_main_per_regime["mid"]
        else:
            base_sigma = float(np.median(sigma_vals))
        regime_penalty_map = {}
        for regime_name, s_val in sigma_main_per_regime.items():
            s = float(s_val) if s_val is not None else base_sigma
            raw = base_sigma / max(s, 1e-8)
            # Clip to a sane range to avoid crazy weights
            regime_penalty_map[regime_name] = float(np.clip(raw, 0.2, 5.0))
    else:
        regime_penalty_map = {name: 1.0 for name in sigma_main_per_regime.keys()}

    last_per_item["regime_penalty"] = last_per_item["regime"].map(regime_penalty_map).fillna(1.0)

    # Composite score: emphasise higher Win %, profit, liquidity, and regime quality
    liq = np.log1p(last_per_item["volume_window"].fillna(0)).clip(lower=1e-3)
    prob = last_per_item["prob_profit"].clip(0.0, 1.0)
    prof = last_per_item["expected_profit"].clip(lower=0.0)

    last_per_item["score"] = (prob ** 2) * prof * liq * last_per_item["regime_penalty"]
    last_per_item.sort_values("score", ascending=False, inplace=True)

    # Load mapping from daily snapshots
    mapping = load_latest_item_mapping(s3, bucket)

    last_per_item["item_id_str"] = last_per_item["item_id"].astype(str)
    mapping["id_str"] = mapping["id"].astype(str)
    last_per_item = last_per_item.merge(
        mapping[["id_str", "name"]],
        left_on="item_id_str",
        right_on="id_str",
        how="left",
    )
    last_per_item.drop(columns=["item_id_str", "id_str"], inplace=True)

    # Persist signals to R2
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")

    signals = []
    for _, r in last_per_item.iterrows():
        signals.append(
            {
                "item_id": int(r["item_id"]),
                "name": r.get("name") or f"Item {int(r['item_id'])}",
                "mid_now": float(r["mid_price"]),
                "future_return_hat": float(r["future_return_hat"]),
                "net_return_hat": float(r["net_return_hat"]),
                "expected_profit": float(r["expected_profit"]),
                "expected_profit_per_second": float(r["expected_profit_per_second"]),
                "prob_profit": float(r["prob_profit"]),
                "volume_window": int(r.get("volume_window") or 0),
                "window_return": float(r.get("window_return") or 0.0),
                "hold_minutes": int(horizon_minutes),
                "regime": str(r.get("regime") or ""),
                "regime_sigma": float(r.get("regime_sigma") or sigma_main_global),
                "regime_penalty": float(r.get("regime_penalty") or 1.0),
                "path": r["path"],
            }
        )

    out = {
        "generated_at_iso": now.isoformat(),
        "horizon_minutes": int(horizon_minutes),
        "path_horizons_minutes": list(path_horizons),
        "tax_rate": float(tax_rate),
        "margin": float(margin),
        "sigma_main": float(sigma_main_global),
        "sigma_main_per_regime": sigma_main_per_regime,
        "regime_defs": regime_defs,
        "signals": signals,
    }

    import json

    buf = json.dumps(out).encode("utf-8")
    
    SIGNALS_NAMESPACE = EXPERIMENT
    key_signals = f"signals/quantile/{date_part}.json"
    key_latest = "signals/quantile/latest.json"
    s3.put_object(Bucket=bucket, Key=key_signals, Body=buf)
    s3.put_object(Bucket=bucket, Key=key_latest, Body=buf)

    print(
        "Using daily snapshot for mapping:",
        mapping.attrs.get("source_key", "unknown"),
    )
    print(f"Loaded {len(mapping)} item names from mapping.")
    print(
        f"Wrote {len(signals)} signals (items with snapshots in the last {WINDOW_MINUTES} minutes)."
    )


if __name__ == "__main__":
    main()
