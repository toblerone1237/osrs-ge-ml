import os
import json
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from features import get_r2_client, list_keys_with_prefix

# How far back to build history (in days)
MAX_HISTORY_DAYS = 14

# Keep full 5m resolution for this many hours, then downsample older data
RECENT_WINDOW_HOURS = 24

# Resolution (minutes) for older history
OLDER_INTERVAL_MIN = 30


def list_recent_5m_keys(s3, bucket: str):
    """
    Collect all 5m snapshot keys for the last MAX_HISTORY_DAYS days.
    """
    now = datetime.now(timezone.utc)
    keys = []
    for delta in range(MAX_HISTORY_DAYS):
        d = (now - timedelta(days=delta)).date()
        prefix = f"5m/{d.year}/{d.month:02d}/{d.day:02d}/"
        day_keys = list_keys_with_prefix(s3, bucket, prefix)
        if day_keys:
            keys.extend(day_keys)

    keys = sorted(set(keys))
    return keys


def build_streaming_histories(s3, bucket: str):
    """
    Stream over 5m snapshots and build per-item histories in memory:

      - last RECENT_WINDOW_HOURS hours: keep every 5m mid price
      - older data up to MAX_HISTORY_DAYS: keep last price in each
        OLDER_INTERVAL_MIN-minute bucket
    """
    now = datetime.now(timezone.utc)
    cutoff_oldest = now - timedelta(days=MAX_HISTORY_DAYS)
    cutoff_recent = now - timedelta(hours=RECENT_WINDOW_HOURS)
    cutoff_oldest_unix = int(cutoff_oldest.timestamp())
    cutoff_recent_unix = int(cutoff_recent.timestamp())

    print("Building histories up to", MAX_HISTORY_DAYS, "days back.")
    print("Recent window cutoff:", cutoff_recent.isoformat())

    # item_id -> list[(ts_unix, mid_price)] for the recent (5m) region
    recent_points = defaultdict(list)

    # item_id -> {bucket_start_ts_unix: mid_price} for the older (30m) region
    older_buckets = defaultdict(dict)

    keys = list_recent_5m_keys(s3, bucket)
    print("Using", len(keys), "5m snapshots in total.")

    for i, key in enumerate(keys, start=1):
        obj = s3.get_object(Bucket=bucket, Key=key)
        snap = json.loads(obj["Body"].read())
        five = snap.get("five_minute") or {}

        ts_unix = int(five.get("timestamp") or 0)
        if ts_unix <= 0:
            continue
        if ts_unix < cutoff_oldest_unix:
            continue

        data = five.get("data") or {}

        if ts_unix >= cutoff_recent_unix:
            # Recent window: keep every 5m value
            for item_id_str, st in data.items():
                ah = st.get("avgHighPrice")
                al = st.get("avgLowPrice")
                if ah is None or al is None:
                    continue
                mid = (ah + al) / 2.0
                if mid is None or mid <= 0 or not math.isfinite(mid):
                    continue
                try:
                    item_id = int(item_id_str)
                except ValueError:
                    continue
                recent_points[item_id].append((ts_unix, float(mid)))
        else:
            # Older region: downsample into OLDER_INTERVAL_MIN-minute buckets
            bucket_sec = (
                ts_unix // (OLDER_INTERVAL_MIN * 60)
            ) * (OLDER_INTERVAL_MIN * 60)
            for item_id_str, st in data.items():
                ah = st.get("avgHighPrice")
                al = st.get("avgLowPrice")
                if ah is None or al is None:
                    continue
                mid = (ah + al) / 2.0
                if mid is None or mid <= 0 or not math.isfinite(mid):
                    continue
                try:
                    item_id = int(item_id_str)
                except ValueError:
                    continue
                # Overwrite within the same bucket: we keep the last 5m price
                older_buckets[item_id][bucket_sec] = float(mid)

        if i % 200 == 0:
            print(f"... processed {i} snapshots")

    return older_buckets, recent_points


def write_histories(s3, bucket: str, older_buckets, recent_points):
    """
    Combine older_buckets & recent_points into full histories
    and write history/{item_id}.json objects to R2.
    """
    now_unix = int(datetime.now(timezone.utc).timestamp())
    min_allowed_unix = now_unix - MAX_HISTORY_DAYS * 24 * 3600

    all_item_ids = set(older_buckets.keys()) | set(recent_points.keys())
    print("Building histories for", len(all_item_ids), "items.")

    written = 0

    for item_id in sorted(all_item_ids):
        pts = []

        # Older (30m) buckets
        older = older_buckets.get(item_id)
        if older:
            for ts, price in older.items():
                if ts >= min_allowed_unix:
                    pts.append((ts, price))

        # Recent 5m points
        recent = recent_points.get(item_id)
        if recent:
            for ts, price in recent:
                if ts >= min_allowed_unix:
                    pts.append((ts, price))

        if not pts:
            continue

        # Sort and de-duplicate by timestamp (keep the last price per ts)
        pts.sort(key=lambda t: t[0])
        dedup = []
        last_ts = None
        for ts, price in pts:
            if last_ts is not None and ts == last_ts:
                dedup[-1] = (ts, price)
            else:
                dedup.append((ts, price))
                last_ts = ts

        history_entries = []
        for ts, price in dedup:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            iso = dt.isoformat().replace("+00:00", "Z")
            history_entries.append(
                {
                    "timestamp_iso": iso,
                    "price": float(price),
                }
            )

        if not history_entries:
            continue

        out = {
            "item_id": int(item_id),
            "history": history_entries,
        }
        body = json.dumps(out).encode("utf-8")
        key = f"history/{int(item_id)}.json"

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

    older_buckets, recent_points = build_streaming_histories(s3, bucket)
    write_histories(s3, bucket, older_buckets, recent_points)


if __name__ == "__main__":
    main()
