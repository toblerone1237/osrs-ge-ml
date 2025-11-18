import os
import json
import io
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
    add_basic_features,
)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# How far back we are willing to look for a "latest" snapshot per item
# when generating signals. Any item that traded at least once in the last
# SCORING_LOOKBACK_MINUTES minutes can get a signal, even if it did not
# appear in the very last 5m bucket.
SCORING_LOOKBACK_MINUTES = 60


# --------------------------------------------------------------------
# Helpers to find recent 5m snapshots and to load models + item map
# --------------------------------------------------------------------


def get_latest_5m_key(s3, bucket: str) -> str | None:
    """
    Find the latest 5m snapshot key by checking today, then yesterday.
    """
    now = datetime.now(timezone.utc)
    for delta in (0, 1):
        d = (now - timedelta(days=delta)).date()
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if keys:
            keys.sort()
            return keys[-1]
    return None


def get_recent_5m_keys(s3, bucket: str, minutes: int) -> list[str]:
    """
    Return all 5m snapshot keys whose timestamp falls within the last
    `minutes` minutes. We only need to look at today + yesterday because
    minutes is assumed < 24h.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=minutes)
    keys_with_ts: list[tuple[datetime, str]] = []

    for delta in (0, 1):  # today and yesterday
        day = (now - timedelta(days=delta)).date()
        prefix = f"5m/{day.year}/{day.month:02d}/{day.day:02d}/"
        day_keys = list_keys_with_prefix(s3, bucket, prefix)
        for key in day_keys:
            # Keys look like: 5m/YYYY/MM/DD/HH-MM.json
            try:
                filename = key.split("/")[-1]
                hhmm = filename.split(".")[0]
                hh_str, mm_str = hhmm.split("-")
                hh = int(hh_str)
                mm = int(mm_str)
                ts = datetime(
                    day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc
                )
            except Exception:
                continue
            if ts >= cutoff:
                keys_with_ts.append((ts, key))

    keys_with_ts.sort(key=lambda x: x[0])
    return [k for _, k in keys_with_ts]


def load_models_and_meta(s3, bucket: str):
    """
    Load multi-horizon regressors + classifier + meta from R2.

    Expected layout (from updated train_model.py):
      - models/xgb/latest_reg.pkl     => either:
            * dict[horizon_minutes] -> XGBRegressor
            * or dict with key "regs" containing that mapping
      - models/xgb/latest_cls.pkl     => XGBClassifier
      - models/xgb/latest_meta.json   => metadata dict
    """
    reg_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_reg.pkl")
    cls_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_cls.pkl")
    meta_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_meta.json")

    reg_bytes = reg_obj["Body"].read()
    cls_bytes = cls_obj["Body"].read()

    reg_bundle = joblib.load(io.BytesIO(reg_bytes))
    cls = joblib.load(io.BytesIO(cls_bytes))
    meta = json.loads(meta_obj["Body"].read())

    # Normalise to a dict[horizon] -> regressor
    if hasattr(reg_bundle, "predict"):
        # Single model only – treat as dict with one horizon (main horizon).
        main_h = int(meta.get("horizon_minutes", 60))
        reg_by_horizon = {main_h: reg_bundle}
    elif isinstance(reg_bundle, dict):
        # Could be {h: reg} or {"regs": {h: reg}, ...}
        if "regs" in reg_bundle and isinstance(reg_bundle["regs"], dict):
            reg_by_horizon = reg_bundle["regs"]
        else:
            reg_by_horizon = reg_bundle
    else:
        raise TypeError("Unsupported reg_bundle type: %r" % type(reg_bundle))

    return reg_by_horizon, cls, meta


def _find_mapping_list(obj):
    """
    Recursively search a nested JSON object for a list of dicts that look
    like OSRS wiki 'mapping' entries (must have 'id' and 'name').
    """
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "id" in obj[0] and "name" in obj[0]:
            return obj
        for el in obj:
            res = _find_mapping_list(el)
            if res is not None:
                return res
    elif isinstance(obj, dict):
        for v in obj.values():
            res = _find_mapping_list(v)
            if res is not None:
                return res
    return None


def load_item_name_map(s3, bucket: str) -> dict[int, str]:
    """
    Load the latest daily snapshot and build {item_id: name} from the wiki
    mapping it contains.
    """
    now = datetime.now(timezone.utc)
    daily_key = None

    # Try today and up to 6 days back
    for delta in range(0, 7):
        d = (now - timedelta(days=delta)).date()
        prefix = f"daily/{d.year}/{d.month:02d}/{d.day:02d}"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = resp.get("Contents")
        if contents:
            contents.sort(key=lambda x: x["Key"])
            daily_key = contents[-1]["Key"]
            break

    if not daily_key:
        print("No daily snapshot found for mapping.")
        return {}

    print("Using daily snapshot for mapping:", daily_key)
    obj = s3.get_object(Bucket=bucket, Key=daily_key)
    daily = json.loads(obj["Body"].read())

    mapping_list = _find_mapping_list(daily)
    if mapping_list is None:
        print("No mapping list with id+name found inside daily snapshot.")
        return {}

    id_to_name: dict[int, str] = {}
    for entry in mapping_list:
        try:
            item_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        name = entry.get("name")
        if name:
            id_to_name[item_id] = name

    print(f"Loaded {len(id_to_name)} item names from mapping.")
    return id_to_name


def build_scoring_dataframe(s3, bucket: str, lookback_minutes: int) -> pd.DataFrame:
    """
    Build a DataFrame of the most recent snapshot per item within the last
    `lookback_minutes` minutes.

    If there are no recent keys, we fall back to the single latest 5m snapshot
    (old behaviour).
    """
    recent_keys = get_recent_5m_keys(s3, bucket, lookback_minutes)
    if not recent_keys:
        print(
            f"No 5m snapshots found in the last {lookback_minutes} minutes; "
            "falling back to latest single snapshot."
        )
        latest_key = get_latest_5m_key(s3, bucket)
        if not latest_key:
            return pd.DataFrame()
        print("Using fallback snapshot:", latest_key)
        df = flatten_5m_snapshots(s3, bucket, [latest_key])
    else:
        print(
            f"Using {len(recent_keys)} snapshots from the last "
            f"{lookback_minutes} minutes."
        )
        df = flatten_5m_snapshots(s3, bucket, recent_keys)

    if df.empty:
        return df

    # Basic features
    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    # For scoring, we want exactly one "current" row per item: the most recent
    # snapshot in the lookback window. We rely on the timestamp column exposed
    # by flatten_5m_snapshots.
    df.sort_values(["item_id", "timestamp"], inplace=True)
    idx = df.groupby("item_id")["timestamp"].idxmax()
    df = df.loc[idx].copy()

    # Very important: make index 0..N-1 so prediction arrays align with row positions
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------
# Main scoring
# --------------------------------------------------------------------


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    # 1) Build scoring dataframe for the last SCORING_LOOKBACK_MINUTES minutes
    df = build_scoring_dataframe(s3, bucket, SCORING_LOOKBACK_MINUTES)
    if df.empty:
        print("No data to score (no recent snapshots).")
        return

    print(f"Scoring {len(df)} items with recent activity.")

    # 2) Load models + meta
    reg_by_horizon, cls, meta = load_models_and_meta(s3, bucket)

    feature_cols = meta["feature_cols"]
    tax = float(meta.get("tax_rate", 0.02))
    main_horizon = int(meta.get("horizon_minutes", 60))

    # Horizons for the path (e.g. [5,10,...,120])
    path_horizons = meta.get("path_horizons_minutes")
    if not path_horizons:
        # Fall back to any keys we can parse as ints
        keys = []
        for k in reg_by_horizon.keys():
            try:
                keys.append(int(k))
            except Exception:
                continue
        path_horizons = sorted(set(keys))

    print("Main horizon:", main_horizon, "minutes.")
    print("Path horizons:", path_horizons)

    # 3) Build feature matrix
    X = df[feature_cols].values

    # 4) Predict returns for each horizon
    def _get_reg(h):
        if h in reg_by_horizon:
            return reg_by_horizon[h]
        if str(h) in reg_by_horizon:
            return reg_by_horizon[str(h)]
        return None

    path_preds: dict[int, np.ndarray] = {}
    for h in path_horizons:
        reg_h = _get_reg(h)
        if reg_h is None:
            print(f"[warn] No regressor for horizon {h}m in model dict.")
            continue
        print(f"[score] Predicting horizon {h}m for {len(df)} rows.")
        path_preds[h] = reg_h.predict(X)

    # Main horizon: use its predictions if present, otherwise first available
    main_ret = path_preds.get(main_horizon)
    if main_ret is None:
        print(
            f"[warn] main_horizon {main_horizon}m not found in path_preds; "
            f"using first available reg instead."
        )
        if path_preds:
            some_h = next(iter(path_preds.keys()))
            main_ret = path_preds[some_h]
        else:
            # Extreme fallback: pick some regressor and ignore horizon mismatch
            some_reg = next(iter(reg_by_horizon.values()))
            main_ret = some_reg.predict(X)

    # 5) Probability of profit from classifier (main horizon)
    try:
        prob_profit = cls.predict_proba(X)[:, 1]
    except Exception as e:
        print("[warn] classifier predict_proba failed:", e)
        prob_profit = np.full(len(df), 0.5, dtype=float)

    df["future_return_hat"] = main_ret
    df["prob_profit"] = prob_profit

    mid = df["mid_price"].values.astype(float)
    buy_price = mid
    mid_future = buy_price * (1.0 + main_ret)
    net_sell = mid_future * (1.0 - tax)
    profit = net_sell - buy_price
    hold_seconds = main_horizon * 60.0
    profit_per_sec = profit / hold_seconds

    df["expected_profit"] = profit
    df["expected_profit_per_second"] = profit_per_sec

    # 6) Load item -> name map
    id_to_name = load_item_name_map(s3, bucket)

    # 7) Build multi-horizon path predictions per row (ALL items)
    signals = []
    N = len(df)
    path_horizons_sorted = sorted(path_horizons)

    for i in range(N):
        row = df.iloc[i]
        item_id = int(row["item_id"])
        name = id_to_name.get(item_id)

        # For each horizon we stored predictions for,
        # grab the i-th prediction from path_preds[h].
        path = []
        for h in path_horizons_sorted:
            preds_h = path_preds.get(h)
            if preds_h is None:
                continue
            # preds_h is length N, indexed 0..N-1
            r = float(preds_h[i])
            path.append(
                {
                    "minutes": int(h),
                    "future_return_hat": r,
                }
            )

        signals.append(
            {
                "item_id": item_id,
                "name": name,  # may be None if mapping doesn't have it
                "mid_now": float(row["mid_price"]),
                "future_return_hat": float(row["future_return_hat"]),
                "prob_profit": float(row["prob_profit"]),
                "expected_profit": float(row["expected_profit"]),
                "expected_profit_per_second": float(
                    row["expected_profit_per_second"]
                ),
                "hold_minutes": int(main_horizon),
                "path": path,
            }
        )

    # 8) Sort signals by expected_profit_per_second (UI still picks top 10)
    signals.sort(
        key=lambda s: s.get("expected_profit_per_second", 0.0),
        reverse=True,
    )

    # 9) Write signals/latest.json + dated copy
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")
    time_part = now.strftime("%H-%M")

    out = {
        "generated_at_iso": now.isoformat(),
        "horizon_minutes": int(main_horizon),
        "tax_rate": float(tax),
        "path_horizons_minutes": [int(h) for h in path_horizons_sorted],
        "signals": signals,
    }
    body = json.dumps(out).encode("utf-8")

    latest_key = "signals/latest.json"
    dated_key = f"signals/{date_part}/{time_part}.json"

    s3.put_object(
        Bucket=bucket,
        Key=latest_key,
        Body=body,
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket,
        Key=dated_key,
        Body=body,
        ContentType="application/json",
    )

    print(
        f"Wrote {len(signals)} signals "
        f"(items with snapshots in the last {SCORING_LOOKBACK_MINUTES} minutes)."
    )


if __name__ == "__main__":
    main()
