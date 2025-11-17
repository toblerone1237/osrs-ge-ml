import os
import json
from datetime import datetime, timezone

import boto3
from botocore.config import Config
import pandas as pd
import numpy as np

# --- Multi‑horizon configuration -------------------------------------------

# Horizons in minutes for which we will train path models
# e.g. 5, 10, ..., 120 minutes
HORIZONS_MINUTES = list(range(5, 125, 5))


def get_r2_client():
    endpoint = os.environ["R2_ENDPOINT"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def list_keys_with_prefix(s3, bucket, prefix):
    """
    List all object keys in an R2 bucket that start with the given prefix.
    """
    keys = []
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            continuation = resp["NextContinuationToken"]
        else:
            break
    return keys


def load_5m_snapshot(s3, bucket, key):
    """
    Load a single 5‑minute snapshot JSON from R2.
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    snap = json.loads(obj["Body"].read())
    return snap


def flatten_5m_snapshots(s3, bucket, keys):
    """
    Flatten a list of OSRS 5‑minute snapshot objects into a tabular DataFrame.

    Columns:
      - timestamp: timezone‑aware datetime (UTC) of the 5m bucket
      - timestamp_unix: integer seconds since epoch (UTC)
      - item_id
      - avg_high_price, avg_low_price
      - high_volume, low_volume
    """
    rows = []
    for key in keys:
        snap = load_5m_snapshot(s3, bucket, key)
        five = snap["five_minute"]
        ts_unix = five["timestamp"]            # seconds since epoch (int)
        ts_dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
        data = five["data"]

        for item_id_str, st in data.items():
            try:
                item_id = int(item_id_str)
            except ValueError:
                continue
            rows.append(
                {
                    "timestamp": ts_dt,
                    "timestamp_unix": ts_unix,
                    "item_id": item_id,
                    "avg_high_price": st.get("avgHighPrice"),
                    "avg_low_price": st.get("avgLowPrice"),
                    "high_volume": st.get("highPriceVolume"),
                    "low_volume": st.get("lowPriceVolume"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.sort_values(["item_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_basic_features(df):
    """
    Add basic per‑snapshot features used by both training and scoring.

    Adds:
      - mid_price
      - spread, spread_pct
      - total_volume_5m
      - log_volume_5m
    """
    df = df.copy()
    df["mid_price"] = (df["avg_high_price"] + df["avg_low_price"]) / 2

    df["spread"] = df["avg_high_price"] - df["avg_low_price"]
    # avoid division by zero
    df["spread_pct"] = df["spread"] / df["mid_price"].replace(0, np.nan)

    df["total_volume_5m"] = (df["high_volume"] + df["low_volume"]).fillna(0)
    df["log_volume_5m"] = np.log1p(df["total_volume_5m"])

    return df


def add_multi_horizon_returns(
    df,
    horizons_minutes=None,
    item_col="item_id",
    time_col="timestamp_unix",
    mid_col="mid_price",
):
    """
    For each row (item_id, timestamp), compute future returns at multiple horizons.

    For each horizon h in `horizons_minutes` (minutes), we compute:

        ret_hm = (mid(t + h) / mid(t)) - 1

    where mid(t+h) is the mid_price at the first snapshot whose timestamp
    is >= t + h minutes (within the same item_id series).

    This function assumes:
      - df[time_col] is integer seconds since epoch (e.g. timestamp_unix)
      - df[mid_col] holds the mid price at that time
      - df is already at 5‑minute resolution per item_id (your 5m snapshots)

    It returns a *copy* of df with new columns:

        ret_5m, ret_10m, ..., ret_120m

    (or whatever horizons you pass in).
    """
    if horizons_minutes is None:
        horizons_minutes = HORIZONS_MINUTES

    df = df.copy()

    # ensure sorted by item + time
    df.sort_values([item_col, time_col], inplace=True)

    # prepare empty columns
    for h in horizons_minutes:
        df[f"ret_{h}m"] = np.nan

    # group per item to keep lookup local
    for item_id, g in df.groupby(item_col):
        idx = g.index.to_numpy()
        ts = g[time_col].to_numpy(dtype="int64")   # seconds
        mid = g[mid_col].to_numpy(dtype="float64")

        n = len(ts)
        if n == 0:
            continue

        for i in range(n):
            mid_now = mid[i]
            if not np.isfinite(mid_now) or mid_now <= 0:
                continue

            t0 = ts[i]

            for h in horizons_minutes:
                target = t0 + h * 60  # seconds

                # find first index j with ts[j] >= target
                j = np.searchsorted(ts, target)
                if j >= n:
                    continue  # no future snapshot available at this horizon

                mid_future = mid[j]
                if not np.isfinite(mid_future) or mid_future <= 0:
                    continue

                ret = (mid_future / mid_now) - 1.0
                df.loc[idx[i], f"ret_{h}m"] = ret

    return df
