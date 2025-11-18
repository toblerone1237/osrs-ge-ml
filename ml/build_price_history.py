import os
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from features import (
    get_r2_client,
    list_keys_with_prefix,
    flatten_5m_snapshots,
    add_basic_features,
)

# How far back to build history (in days)
MAX_HISTORY_DAYS = 14

# Keep full 5m resolution for this many hours, then downsample older data
RECENT_WINDOW_HOURS = 24

# Resolution (minutes) for older history
OLDER_INTERVAL_MIN = 30


def find_recent_5m_keys(s3, bucket: str) -> list[str]:
    """
    Collect all 5m snapshot keys for the last MAX_HISTORY_DAYS days.
    """
    now = datetime.now(timezone.utc)
    keys: list[str] = []

    for delta in range(MAX_HISTORY_DAYS):
        d = (now - timedelta(days=delta)).date()
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        day_keys = list_keys_with_prefix(s3, bucket, prefix)
        if day_keys:
            keys.extend(day_keys)

    keys = sorted(set(keys))
    return keys


def build_histories(df: pd.DataFrame, bucket: str, s3) -> None:
    """
    Given a DataFrame of 5m snapshots with mid_price, build per-item histories and
    write them to R2 as history/{item_id}.json.

    - Last RECENT_WINDOW_HOURS hours stay at 5m resolution
    - Older data is resampled to OLDER_INTERVAL_MIN-minute buckets (last price)
    """
    if df.empty:
        print("No rows to build histories from.")
        return

    # Ensure timestamp is a timezone-aware index
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    cutoff_recent = datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)
    print("Recent/older split cutoff:", cutoff_recent.isoformat())

    n_items = df["item_id"].nunique()
    print(f"Building histories for {n_items} items.")

    written = 0

    for item_id, g in df.groupby("item_id"):
        # g already indexed by timestamp thanks to set_index above
        g = g.sort_index()

        older = g[g.index < cutoff_recent]
        recent = g[g.index >= cutoff_recent]

        pieces = []

        if not older.empty:
            # Downsample older region to 30m buckets (last mid_price in each bucket)
            series = older["mid_price"]
            older_30m = (
                series.resample(f"{OLDER_INTERVAL_MIN}min")  # <-- 'min', not 'T'
                .last()
                .dropna()
            )
            if not older_30m.empty:
                older_df = older_30m.reset_index()
                older_df.rename(columns={"mid_price": "price"}, inplace=True)
                pieces.append(older_df)

        if not recent.empty:
            # Keep 5m resolution in recent region
            recent_df = recent.reset_index()[["timestamp", "mid_price"]]
            recent_df.rename(columns={"mid_price": "price"}, inplace=True)
            pieces.append(recent_df)

        # Avoid concatenating empties → removes the FutureWarning
        if not pieces:
            continue

        combined = pd.concat(pieces, ignore_index=True)
        combined.sort_values("timestamp", inplace=True)
        combined.drop_duplicates(subset="timestamp", keep="last", inplace=True)

        # Convert to JSON-serialisable list
        history_entries = []
        for ts, price in zip(combined["timestamp"], combined["price"]):
            if pd.isna(price) or not np.isfinite(price):
                continue
            ts_iso = ts.isoformat().replace("+00:00", "Z")
            history_entries.append(
                {
                    "timestamp_iso": ts_iso,
                    "price": float(price),
                }
            )

        if not history_entries:
            continue

        out = {
            "item_id": int(item_id),
            "history": history_entries,
        }
        key = f"history/{int(item_id)}.json"
        body = json.dumps(out).encode("utf-8")

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

        written += 1
        if written % 200 == 0:
            print(f"... wrote history for {written} items so far")

    print(f"Done. Wrote history for {written} items.")


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    # 1) Collect relevant 5m snapshot keys
    keys = find_recent_5m_keys(s3, bucket)
    if not keys:
        print(f"No 5m snapshots found in last {MAX_HISTORY_DAYS} days.")
        return

    print(f"Using {len(keys)} 5m snapshots over last {MAX_HISTORY_DAYS} days.")

    # 2) Flatten into a DataFrame and compute mid_price
    df = flatten_5m_snapshots(s3, bucket, keys)
    if df.empty:
        print("Flattened DataFrame is empty.")
        return

    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"])
    df = df[df["mid_price"] > 0].copy()

    # 3) Build and upload per-item histories
    build_histories(df, bucket, s3)


if __name__ == "__main__":
    main()
