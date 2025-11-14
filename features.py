import os
import json
from datetime import datetime, timezone

import boto3
from botocore.config import Config
import pandas as pd
import numpy as np


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
    obj = s3.get_object(Bucket=bucket, Key=key)
    snap = json.loads(obj["Body"].read())
    return snap


def flatten_5m_snapshots(s3, bucket, keys):
    rows = []
    for key in keys:
        snap = load_5m_snapshot(s3, bucket, key)
        five = snap["five_minute"]
        ts = five["timestamp"]
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        data = five["data"]

        for item_id_str, st in data.items():
            try:
                item_id = int(item_id_str)
            except ValueError:
                continue
            rows.append(
                {
                    "timestamp": ts_dt,
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
    df = df.copy()
    df["mid_price"] = (df["avg_high_price"] + df["avg_low_price"]) / 2
    df["spread"] = df["avg_high_price"] - df["avg_low_price"]
    df["spread_pct"] = df["spread"] / df["mid_price"].replace(0, np.nan)
    df["total_volume_5m"] = (df["high_volume"] + df["low_volume"]).fillna(0)
    df["log_volume_5m"] = np.log1p(df["total_volume_5m"])
    return df
