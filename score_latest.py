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
# Helpers to find the latest 5m snapshot and to load models + item map
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


def load_models_and_meta(s3, bucket: str):
    """
    Load multi-horizon regressors + classifier + meta from R2.

    Assumes:
      - models/xgb/latest_reg.pkl     => dict[horizon_minutes] -> XGBRegressor
      - models/xgb/latest_cls.pkl     => XGBClassifier
      - models/xgb/latest_meta.json   => metadata dict
    """
    reg_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_reg.pkl")
    cls_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_cls.pkl")
    meta_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_meta.json")

    reg_bytes = reg_obj["Body"].read()
    cls_bytes = cls_obj["Body"].read()

    reg_by_horizon = joblib.load(io.BytesIO(reg_bytes))
    cls = joblib.load(io.BytesIO(cls_bytes))
    meta = json.loads(meta_obj["Body"].read())

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


# --------------------------------------------------------------------
# Main scoring
# --------------------------------------------------------------------


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    # 1) Get the latest 5m snapshot and flatten it
    latest_key = get_latest_5m_key(s3, bucket)
    if not latest_key:
        print("No 5m snapshot found.")
        return

    print("Using snapshot:", latest_key)
    df = flatten_5m_snapshots(s3, bucket, [latest_key])
    if df.empty:
        print("Snapshot empty.")
        return

    # 2) Basic features + clean
    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    # At this point we expect:
    #   columns: timestamp, item_id, avg_high_price, avg_low_price,
    #   high_volume, low_volume, mid_price, spread, spread_pct,
    #   total_volume_5m, log_volume_5m

    # 3) Load models + meta
    reg_by_horizon, cls, meta = load_models_and_meta(s3, bucket)

    feature_cols = meta["feature_cols"]
    tax = float(meta.get("tax_rate", 0.02))
    main_horizon = int(meta.get("horizon_minutes", 60))

    # Horizons for the path (list of minutes, e.g. [5,10,...,120])
    path_horizons = meta.get("path_horizons_minutes")
    if not path_horizons:
        # fall back to whatever keys the reg dict has
        # (keys might be ints or strings)
        keys = []
        for k in reg_by_horizon.keys():
            try:
                keys.append(int(k))
            except Exception:
                continue
        path_horizons = sorted(set(keys))

    print("Main horizon:", main_horizon, "minutes.")
    print("Path horizons:", path_horizons)

    # 4) Build feature matrix
    X = df[feature_cols].values

    # 5) Predict returns for each horizon
    # reg_by_horizon keys may be ints or strings; support both.
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

    # Main horizon: use its predictions if present, otherwise whatever reg is available
    main_ret = path_preds.get(main_horizon)
    if main_ret is None:
        print(f"[warn] main_horizon {main_horizon}m not found in path_preds; "
              f"using first available reg instead.")
        # Pick the first available
        if path_preds:
            some_h = next(iter(path_preds.keys()))
            main_ret = path_preds[some_h]
        else:
            # Absolute fallback: try any reg in dict
            some_reg = next(iter(reg_by_horizon.values()))
            main_ret = some_reg.predict(X)

    # 6) Probability of profit from classifier (main horizon)
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

    # 7) Load item -> name map
    id_to_name = load_item_name_map(s3, bucket)

    # 8) Build multi-horizon path predictions per row
    # path_preds[h] is a numpy array aligned with df.index
    signals = []

    # For convenience when iterating
    path_horizons_sorted = sorted(path_horizons)

    for idx, row in df.iterrows():
        item_id = int(row["item_id"])
        name = id_to_name.get(item_id)

        path = []
        for h in path_horizons_sorted:
            preds_h = path_preds.get(h)
            if preds_h is None:
                continue
            r = float(preds_h[idx])
            path.append(
                {
                    "minutes": int(h),
                    "future_return_hat": r,
                }
            )

        signals.append(
            {
                "item_id": item_id,
                "name": name,  # may be None if not found in mapping
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

    # Sort signals mainly for convenience (UI resorts anyway)
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

    print(f"Wrote {len(signals)} signals (all items in latest snapshot).")


if __name__ == "__main__":
    main()
