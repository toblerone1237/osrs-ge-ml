import os
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
)

MAX_DAYS = 14
LAST_24H_HOURS = 24
OLDER_INTERVAL_MIN = 30  # minutes

def list_5m_keys_last_days(s3, bucket, days=MAX_DAYS):
    now = datetime.now(timezone.utc)
    keys = []
    for delta in range(days):
        d = (now - timedelta(days=delta)).date()
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        keys.extend(list_keys_with_prefix(s3, bucket, prefix))
    keys.sort()
    return keys

def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    keys = list_5m_keys_last_days(s3, bucket, MAX_DAYS)
    if not keys:
        print("No 5m snapshots found.")
        return

    print(f"Loading {len(keys)} 5m snapshots into DataFrame...")
    df = flatten_5m_snapshots(s3, bucket, keys)
    if df.empty:
        print("Empty DataFrame after flatten_5m_snapshots.")
        return

    # mid price
    df["mid_price"] = (df["avg_high_price"] + df["avg_low_price"]) / 2.0
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    # ensure datetime index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    newest_ts = df["timestamp"].max()
    cutoff_recent = newest_ts - timedelta(hours=LAST_24H_HOURS)
    cutoff_oldest = newest_ts - timedelta(days=MAX_DAYS)

    print("Newest timestamp:", newest_ts.isoformat())
    print("Recent cutoff:", cutoff_recent.isoformat())
    print("Oldest cutoff:", cutoff_oldest.isoformat())

    # We only care about the last MAX_DAYS window anyway
    df = df[df["timestamp"] >= cutoff_oldest].copy()

    # Group by item and build history for each one
    grouped = df.groupby("item_id", sort=False)

    for item_id, g in grouped:
        g = g.sort_values("timestamp")

        recent = g[g["timestamp"] >= cutoff_recent].copy()
        older = g[g["timestamp"] < cutoff_recent].copy()

        # recent: keep all 5m points (as-is)
        recent_history = recent[["timestamp", "mid_price"]]

        # older: resample to 30m by last observed price in each 30m bin
        if not older.empty:
            older = older.set_index("timestamp")
            older_30m = older["mid_price"].resample(f"{OLDER_INTERVAL_MIN}T").last().dropna()
            older_history = older_30m.reset_index()
        else:
            older_history = pd.DataFrame(columns=["timestamp", "mid_price"])

        combined = pd.concat([older_history, recent_history], ignore_index=True)
        if combined.empty:
            continue

        # Build compact history list
        history = []
        for _, row in combined.iterrows():
            ts = row["timestamp"]
            price = float(row["mid_price"])
            history.append(
                {
                    "timestamp_iso": ts.isoformat(),
                    "price": price,
                }
            )

        out = {
            "item_id": int(item_id),
            "history": history,
        }
        body = json.dumps(out).encode("utf-8")
        key = f"history/{int(item_id)}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

        # Optional: print or limit number of items for testing
        # print(f"Wrote history for item {item_id} with {len(history)} points")

    print("Done building per-item histories.")

if __name__ == "__main__":
    main()
