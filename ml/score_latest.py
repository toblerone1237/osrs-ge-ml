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
    HORIZONS_MINUTES,  # [5, 10, ..., 120]
)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

HORIZON_MINUTES = 60   # main decision horizon (must be in HORIZONS_MINUTES)
WINDOW_MINUTES = 60    # how far back to pull snapshots for scoring
TAX_RATE = 0.02        # same as in training
MARGIN = 0.002         # same as in training

# Minimum volume over the scoring window for an item to be considered "active"
MIN_VOLUME_WINDOW = 1

# Clamp predictions before converting to EV/Win%
FUTURE_RETURN_HAT_CLIP_LOWER = -0.5   # -50%
FUTURE_RETURN_HAT_CLIP_UPPER = +0.5   # +50%

# Heuristic "noisy regime" flags
NOISY_PRICE_MAX_GP = 10_000           # cheap items tend to be regime-shifted
NOISY_NET_RETURN_MIN = 0.20           # 20% predicted net return is suspicious here
NOISY_PROB_MIN = 0.80                 # 80%+ Win% on cheap items often unreliable


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


def load_latest_models(s3, bucket):
    """
    Load the latest regressors and metadata from R2.
    """
    key_reg = "models/remove-noisy-sections/xgb/latest_reg.pkl"
    key_meta = "models/remove-noisy-sections/xgb/latest_meta.json"

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
    all_keys.sort()

    # Determine the latest timestamp
    obj_latest = s3.get_object(Bucket=bucket, Key=latest_key)
    import json as _json
    snap_latest = _json.loads(obj_latest["Body"].read())
    latest_ts = snap_latest["five_minute"]["timestamp"]

    window_start_ts = latest_ts - WINDOW_MINUTES * 60

    recent_keys = []
    for key in all_keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        snap = _json.loads(obj["Body"].read())
        ts = snap["five_minute"]["timestamp"]
        if ts >= window_start_ts:
            recent_keys.append(key)

    print(f"Using {len(recent_keys)} snapshots from the last {WINDOW_MINUTES} minutes.")

    df = flatten_5m_snapshots(s3, bucket, recent_keys)
    if df.empty:
        raise RuntimeError("No rows in scoring DataFrame after flatten.")

    df = add_model_features(df)
    df = df[df["mid_price"] > 0].copy()  # sanity only
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
    return df


def normal_cdf_array(z: np.ndarray) -> np.ndarray:
    """
    Vectorised Normal(0,1) CDF using math.erf.
    """
    def _cdf_scalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    vec = np.vectorize(_cdf_scalar, otypes=[float])
    return vec(z)


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    reg_models, meta = load_latest_models(s3, bucket)

    horizon_minutes = int(meta.get("horizon_minutes", HORIZON_MINUTES))
    path_horizons = meta.get("path_horizons_minutes", HORIZONS_MINUTES)
    feature_cols = meta.get("feature_cols", ["mid_price", "spread_pct", "log_volume_5m"])
    tax_rate = float(meta.get("tax_rate", TAX_RATE))
    margin = float(meta.get("margin", MARGIN))
    sigma_main = float(meta.get("sigma_main", 0.02))

    df_raw, latest_ts = build_scoring_dataframe(s3, bucket)
    window_start_ts = latest_ts - WINDOW_MINUTES * 60

    df_agg = compute_window_aggregates(df_raw, window_start_ts)
    df_active = filter_active_items(df_agg)

    # Merge aggregated stats back into the per-snapshot data
    df = df_raw.merge(df_active[["item_id"]], on="item_id", how="inner")
    print(f"Scoring {df['item_id'].nunique()} items with recent activity.")

    X_all = df[feature_cols].values

    # For each path horizon, compute predictions
    path_results = {}
    for H in path_horizons:
        reg = reg_models.get(H)
        if reg is None:
            continue
        print(f"[score] Predicting horizon {H}m for {len(df)} rows.")
        preds = reg.predict(X_all)
        path_results[H] = preds

    # Main horizon predictions
    if horizon_minutes not in reg_models:
        raise RuntimeError(f"Missing regressor for main horizon {horizon_minutes}m.")
    reg_main = reg_models[horizon_minutes]
    main_ret_raw = reg_main.predict(X_all)

    # Clamp before converting to anything downstream
    df["future_return_hat_raw"] = main_ret_raw
    df["future_return_hat"] = np.clip(
        main_ret_raw, FUTURE_RETURN_HAT_CLIP_LOWER, FUTURE_RETURN_HAT_CLIP_UPPER
    )

    # Profit/EV
    df["net_return_hat"] = (1.0 + df["future_return_hat"]) * (1.0 - tax_rate) - 1.0
    df["buy_price"] = df["mid_price"]
    df["sell_price_hat"] = df["buy_price"] * (1.0 + df["net_return_hat"])
    df["expected_profit"] = df["sell_price_hat"] - df["buy_price"]
    hold_seconds = horizon_minutes * 60.0
    df["expected_profit_per_second"] = df["expected_profit"] / hold_seconds

    # Win % via Normal CDF against gross threshold
    gross_threshold = (1.0 + margin) / (1.0 - tax_rate) - 1.0
    z = (df["future_return_hat"].values - gross_threshold) / max(sigma_main, 1e-8)
    prob_profit = normal_cdf_array(z)
    df["prob_profit"] = np.clip(prob_profit, 1e-4, 1.0 - 1e-4)

    # Summarise per item (last snapshot)
    df.sort_values(["item_id", "timestamp"], inplace=True)
    last_per_item = df.groupby("item_id").tail(1).copy()
    last_per_item = last_per_item.merge(df_agg, on="item_id", how="left")

    # Mark "noisy regimes" so downstream (UI, etc.) can exclude or flag them
    reasons = []
    is_cheap = (last_per_item["mid_price"] <= NOISY_PRICE_MAX_GP)
    big_net = (last_per_item["net_return_hat"].abs() >= NOISY_NET_RETURN_MIN)
    high_prob = (last_per_item["prob_profit"] >= NOISY_PROB_MIN)
    noisy_mask = is_cheap & (big_net | high_prob)
    last_per_item["noisy"] = noisy_mask

    reason_text = []
    for _, r in last_per_item.iterrows():
        r_reasons = []
        if r["mid_price"] <= NOISY_PRICE_MAX_GP:
            r_reasons.append(f"mid_price <= {NOISY_PRICE_MAX_GP:,} gp")
        if abs(r["net_return_hat"]) >= NOISY_NET_RETURN_MIN:
            r_reasons.append(f"large predicted net return ({r['net_return_hat']:.0%})")
        if r["prob_profit"] >= NOISY_PROB_MIN:
            r_reasons.append(f"very high Win% estimate ({r['prob_profit']:.0%})")
        reason_text.append("; ".join(r_reasons) if r_reasons else "")
    last_per_item["noise_reason"] = reason_text

    # Build path predictions for each item (at the last snapshot index)
    path_rows = []
    for _, r in last_per_item.iterrows():
        item_id = r["item_id"]
        mask = (df["item_id"] == item_id)
        row_idx = df.index[mask][-1]
        row_path = []
        for h in sorted(path_horizons):
            preds_h = path_results.get(h)
            if preds_h is None:
                continue
            ret_hat = preds_h[row_idx]
            # Clamp for path as well, to be consistent
            ret_hat = float(np.clip(ret_hat, FUTURE_RETURN_HAT_CLIP_LOWER, FUTURE_RETURN_HAT_CLIP_UPPER))
            row_path.append({"minutes": int(h), "future_return_hat": ret_hat})
        path_rows.append(row_path)
    last_per_item["path"] = path_rows

    # Load mapping from daily snapshots for item names
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

    # Composite score: EV × liquidity × mild prob shaping
    liq = np.log1p(last_per_item["volume_window"].fillna(0)).clip(lower=1e-3)
    prob = last_per_item["prob_profit"].clip(0.0, 1.0)
    prof = last_per_item["expected_profit"].clip(lower=0.0)
    ev_profit = prob * prof
    prob_boost = 0.5 + 0.5 * prob  # modestly rewards higher probabilities, saturates at 1.0
    last_per_item["score"] = ev_profit * liq * prob_boost

    # Hard-penalise noisy
    last_per_item.loc[last_per_item["noisy"], "score"] = 0.0

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
                "path": r["path"],
                "score": float(r["score"]),
                "noisy": bool(r["noisy"]),
                "noise_reason": r["noise_reason"],
            }
        )

    out = {
        "generated_at_iso": now.isoformat(),
        "horizon_minutes": int(horizon_minutes),
        "path_horizons_minutes": list(path_horizons),
        "tax_rate": float(tax_rate),
        "margin": float(margin),
        "sigma_main": float(sigma_main),
        # FYI for UI/debug
        "future_return_hat_clip": [FUTURE_RETURN_HAT_CLIP_LOWER, FUTURE_RETURN_HAT_CLIP_UPPER],
        "signals": signals,
    }

    import json
    buf = json.dumps(out).encode("utf-8")

    key_signals = f"signals/{date_part}.json"
    key_latest = "signals/latest.json"
    s3.put_object(Bucket=bucket, Key=key_signals, Body=buf)
    s3.put_object(Bucket=bucket, Key=key_latest, Body=buf)

    print(
        f"Wrote {len(signals)} signals (items with snapshots in the last {WINDOW_MINUTES} minutes)."
    )


def load_latest_item_mapping(s3, bucket):
    """
    Load the latest item mapping from daily snapshots (look back up to 7 days).
    """
    now = datetime.now(timezone.utc)
    import json

    for delta in range(7):
        d = (now - timedelta(days=delta)).date()
        prefix = f"daily/{d.year}/{d.month:02d}/{d.day:02d}"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if not keys:
            continue

        keys.sort()
        latest_key = keys[-1]
        obj = s3.get_object(Bucket=bucket, Key=latest_key)
        snap = json.loads(obj["Body"].read())
        mapping = extract_item_mapping_from_daily(snap)
        mapping.attrs["source_key"] = latest_key
        return mapping

    return pd.DataFrame(columns=["id", "name"])


def extract_item_mapping_from_daily(snap: dict) -> pd.DataFrame:
    """
    Extract an item mapping from a daily snapshot JSON.
    """
    items = snap.get("items")
    if isinstance(items, list) and items:
        df = pd.DataFrame(items)
        if "id" in df.columns and "name" in df.columns:
            return df[["id", "name"]]

    def dfs(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict) and "id" in obj[0] and "name" in obj[0]:
                return pd.DataFrame(obj)
            for el in obj:
                res = dfs(el);  # noqa
                if res is not None:
                    return res
        elif isinstance(obj, dict):
            for v in obj.values():
                res = dfs(v)  # noqa
                if res is not None:
                    return res
        return None

    res = dfs(snap)
    if res is not None and "id" in res.columns and "name" in res.columns:
        return res[["id", "name"]]
    return pd.DataFrame(columns=["id", "name"])


if __name__ == "__main__":
    main()
