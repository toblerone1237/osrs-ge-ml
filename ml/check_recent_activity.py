# ml/check_recent_activity.py

import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
    add_basic_features,
)
from score_latest import get_latest_5m_key


def get_last_12_keys(s3, bucket: str) -> list[str]:
    """
    Get the last ~60 minutes of 5m snapshot keys (up to 12 keys).
    Uses the same naming/ordering assumptions as score_latest.py.
    """
    latest = get_latest_5m_key(s3, bucket)
    if not latest:
        print("No latest 5m snapshot found.")
        return []

    # All snapshots for that day
    day_prefix = latest.rsplit("/", 1)[0] + "/"
    all_keys = list_keys_with_prefix(s3, bucket, day_prefix)
    if not all_keys:
        print("No keys found under prefix", day_prefix)
        return []

    all_keys.sort()
    try:
        idx = all_keys.index(latest)
    except ValueError:
        # Fallback: just use last 12 sorted keys
        return all_keys[-12:]

    start = max(0, idx - 11)
    return all_keys[start : idx + 1]


def main(argv):
    if len(argv) < 2:
        print("Usage: python check_recent_activity.py <item_id> [<item_id> ...]")
        return

    item_ids = [int(x) for x in argv[1:]]

    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    keys = get_last_12_keys(s3, bucket)
    if not keys:
        return

    print("Using these 5m snapshots:")
    for k in keys:
        print("  ", k)

    df = flatten_5m_snapshots(s3, bucket, keys)
    if df.empty:
        print("Flattened DataFrame is empty for the last 60 minutes.")
        return

    df = add_basic_features(df)

    # Combine high+low volume as a basic "was there any trade" proxy
    df["total_volume_5m"] = (df["high_volume"].fillna(0) + df["low_volume"].fillna(0))

    for item_id in item_ids:
        sub = df[df["item_id"] == item_id].copy()
        if sub.empty:
            print(f"\nItem {item_id}: no rows in the last 60 minutes (no snapshot entries).")
            continue

        total_vol = int(sub["total_volume_5m"].sum())
        nonzero_buckets = int((sub["total_volume_5m"] > 0).sum())
        last_ts = sub["timestamp"].max()

        print(f"\nItem {item_id}:")
        print(f"  buckets seen:       {len(sub)}")
        print(f"  non-zero buckets:   {nonzero_buckets}")
        print(f"  total volume (5m):  {total_vol}")
        print(f"  last snapshot time: {last_ts.isoformat()}")

        # Show the last few buckets for debugging
        tail = sub.sort_values("timestamp").tail(5)
        print("  last 5 buckets:")
        for _, r in tail.iterrows():
            print(
                "   ",
                r["timestamp"].isoformat(),
                "mid=",
                r["mid_price"],
                "vol=",
                int(r["total_volume_5m"]),
            )


if __name__ == "__main__":
    main(sys.argv)
