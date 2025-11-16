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


def get_latest_5m_key(s3, bucket):
    """
    Find the latest 5m snapshot key by looking at today then yesterday.
    """
    now = datetime.now(timezone.utc)
    for delta in [0, 1]:
        d = (now - timedelta(days=delta)).date()
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if keys:
            keys.sort()
            return keys[-1]
    return None


def load_models_and_meta(s3, bucket):
    """
    Load the latest regression models (dict of horizon -> regressor),
    the 60m classifier, and meta from R2.
    """
    reg_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_reg.pkl")
    cls_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_cls.pkl")
    meta_obj = s3.get_object(Bucket=bucket, Key="models/xgb/latest_meta.json")

    reg_bytes = reg_obj["Body"].read()
    cls_bytes = cls_obj["Body"].read()

    reg_models = joblib.load(io.BytesIO(reg_bytes))  # dict: {horizon_minutes: model}
    cls = joblib.load(io.BytesIO(cls_bytes))
    meta = json.loads(meta_obj["Body"].read())

    return reg_models, cls, meta


def _find_mapping_list(obj):
    """
    Recursively search a nested JSON object for a list of dicts
    that look like OSRS mapping entries (have 'id' and 'name' keys).
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


def load_item_name_map(s3, bucket):
    """
    Load the latest daily snapshot from R2 and build {item_id: name}.

    - Looks for the most recent daily/YYYY/MM/DD* key (today, then back 7 days).
    - Recursively searches the JSON for a list of objects with 'id' and 'name' keys.
    """
    now = datetime.now(timezone.utc)
    daily_key = None

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

    id_to_name = {}
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


def _get_reg_for_horizon(reg_models, h):
    """
    Robustly fetch the regressor for a given horizon (int minutes),
    handling both int and string keys just in case.
    """
    if h in reg_models:
        return reg_models[h]
    if str(h) in reg_models:
        return reg_models[str(h)]
    return None


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    # 1) latest 5m snapshot
    latest_key = get_latest_5m_key(s3, bucket)
    if not latest_key:
        print("No 5m snapshot found.")
        return

    print("Using snapshot:", latest_key)
    df = flatten_5m_snapshots(s3, bucket, [latest_key])
    if df.empty:
        print("Snapshot empty.")
        return

    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    # 2) load models + meta
    reg_models, cls, meta = load_models_and_meta(s3, bucket)
    feature_cols = meta["feature_cols"]
    H_main = int(meta["horizon_minutes"])
    tax = float(meta["tax_rate"])
    path_horizons = meta.get("path_horizons_minutes", [H_main])
    path_horizons = [int(h) for h in path_horizons]

    # Only keep horizons we actually have models for
    path_horizons = [h for h in path_horizons if _get_reg_for_horizon(reg_models, h) is not None]
    if H_main not in path_horizons:
        path_horizons.append(H_main)
    path_horizons = sorted(set(path_horizons))

    # 3) load item name map from latest daily snapshot
    id_to_name = load_item_name_map(s3, bucket)

    # 4) build features
    X = df[feature_cols].values
    mid = df["mid_price"].values

    # 4a) main horizon prediction (for profit / ranking)
    main_reg = _get_reg_for_horizon(reg_models, H_main)
    if main_reg is None:
        print(f"No regression model found for main horizon {H_main}m.")
        return

    future_ret_main = main_reg.predict(X)
    prob_profit = cls.predict_proba(X)[:, 1]

    # 4b) full path predictions for all horizons
    path_predictions = {}
    for h in path_horizons:
        reg_h = _get_reg_for_horizon(reg_models, h)
        if reg_h is None:
            continue
        path_predictions[h] = reg_h.predict(X)

    # Align predictions with df index for easy lookup
    path_pred_series = {
        h: pd.Series(path_predictions[h], index=df.index) for h in path_predictions
    }

    # 5) profit metrics at main horizon
    df["future_return_hat"] = future_ret_main
    df["prob_profit"] = prob_profit

    mid_future = mid * (1.0 + future_ret_main)
    buy_price = mid
    net_sell = mid_future * (1.0 - tax)
    profit = net_sell - buy_price
    hold_seconds = H_main * 60.0
    profit_per_sec = profit / hold_seconds

    df["expected_profit"] = profit
    df["expected_profit_per_second"] = profit_per_sec

    # 6) filter and rank candidates
    mask = (df["expected_profit"] > 0) & (df["prob_profit"] > 0.55)
    candidates = df[mask].copy()
    candidates.sort_values("expected_profit_per_second", ascending=False, inplace=True)

    # 7) build signals list with item names + full path
    signals = []
    top_n = 200
    for idx, row in candidates.head(top_n).iterrows():
        item_id = int(row["item_id"])
        name = id_to_name.get(item_id)
        mid_now = float(row["mid_price"])

        # Build this item's path from all horizons we have
        path = []
        for h in sorted(path_pred_series.keys()):
            val = path_pred_series[h].loc[idx]
            if not np.isfinite(val):
                continue
            path.append(
                {
                    "minutes": int(h),
                    "future_return_hat": float(val),
                }
            )

        signals.append(
            {
                "item_id": item_id,
                "name": name,  # may be null if mapping doesn't have it
                "mid_now": mid_now,
                "future_return_hat": float(row["future_return_hat"]),  # main horizon
                "prob_profit": float(row["prob_profit"]),
                "expected_profit": float(row["expected_profit"]),
                "expected_profit_per_second": float(row["expected_profit_per_second"]),
                "hold_minutes": int(H_main),
                "path": path,
            }
        )

    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")
    time_part = now.strftime("%H-%M")

    out = {
        "generated_at_iso": now.isoformat(),
        "horizon_minutes": int(H_main),
        "tax_rate": tax,
        "path_horizons_minutes": sorted(path_pred_series.keys()),
        "signals": signals,
    }
    body = json.dumps(out).encode("utf-8")

    latest_key = "signals/latest.json"
    dated_key = f"signals/{date_part}/{time_part}.json"

    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")

    print(f"Wrote {len(signals)} signals.")


if __name__ == "__main__":
    main()
