import os
import json
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from features import get_r2_client, list_keys_with_prefix

# How far back to build history (in days)
MAX_HISTORY_DAYS = 14

# Keep full 5m resolution for this many hours, then downsample older data
RECENT_WINDOW_HOURS = 24

# Resolution (minutes) for older history
OLDER_INTERVAL_MIN = 30

# Number of concurrent fetches for 5m snapshots
SNAPSHOT_FETCH_WORKERS = int(os.environ.get("SNAPSHOT_FETCH_WORKERS", "8"))

# Metadata key to track the last processed snapshot
HISTORY_META_KEY = "history/_meta.json"


def parse_iso_to_unix(ts_iso: str) -> int | None:
    try:
        if ts_iso.endswith("Z"):
            ts_iso = ts_iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_iso)
        return int(dt.timestamp())
    except Exception:
        return None


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


def load_history_meta(s3, bucket: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=HISTORY_META_KEY)
    except Exception:
        return None
    try:
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def save_history_meta(s3, bucket: str, meta: dict):
    body = json.dumps(meta).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=HISTORY_META_KEY, Body=body)


def load_existing_histories(s3, bucket: str, cutoff_oldest_unix: int, cutoff_recent_unix: int):
    """
    Load existing history files to seed older_buckets/recent_points and record
    what is currently stored per item for change detection.
    """
    older_buckets = defaultdict(dict)
    recent_points = defaultdict(list)
    stored_pairs = {}  # item_id -> list[(ts_unix, price)] as stored (no cutoffs)

    history_keys = list_keys_with_prefix(s3, bucket, "history/")
    history_keys = [k for k in history_keys if k.endswith(".json") and k != HISTORY_META_KEY]
    history_keys.sort()

    def load_one(key):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return key, json.loads(obj["Body"].read())
        except Exception:
            return key, None

    with ThreadPoolExecutor(max_workers=SNAPSHOT_FETCH_WORKERS) as pool:
        futures = [pool.submit(load_one, key) for key in history_keys]
        for idx, fut in enumerate(as_completed(futures), start=1):
            key, data = fut.result()
            if data is None:
                continue
            item_id = data.get("item_id")
            if item_id is None:
                # Fallback: try to parse from filename history/{id}.json
                try:
                    item_id = int(os.path.splitext(os.path.basename(key))[0])
                except Exception:
                    continue

            entries = data.get("history") or []
            pairs = []
            for entry in entries:
                price = entry.get("price")
                ts_iso = entry.get("timestamp_iso") or entry.get("timestamp")
                if price is None or ts_iso is None:
                    continue
                try:
                    price_f = float(price)
                except Exception:
                    continue
                if not math.isfinite(price_f) or price_f <= 0:
                    continue
                ts_unix = parse_iso_to_unix(str(ts_iso))
                if ts_unix is None:
                    continue
                pairs.append((ts_unix, price_f))

                if ts_unix < cutoff_oldest_unix:
                    continue
                if ts_unix >= cutoff_recent_unix:
                    recent_points[item_id].append((ts_unix, price_f))
                else:
                    bucket_sec = (ts_unix // (OLDER_INTERVAL_MIN * 60)) * (OLDER_INTERVAL_MIN * 60)
                    older_buckets[item_id][bucket_sec] = price_f

            pairs.sort(key=lambda t: t[0])
            dedup_pairs = []
            last_ts = None
            for ts, price in pairs:
                if last_ts is not None and ts == last_ts:
                    dedup_pairs[-1] = (ts, price)
                else:
                    dedup_pairs.append((ts, price))
                    last_ts = ts
            stored_pairs[item_id] = dedup_pairs

            if idx % 500 == 0:
                print(f"... loaded {idx} existing history files")

    return older_buckets, recent_points, stored_pairs


def fetch_snapshot(s3, bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    snap = json.loads(obj["Body"].read())
    five = snap.get("five_minute") or {}
    return five


def apply_new_snapshots(
    s3,
    bucket: str,
    keys: list[str],
    cutoff_oldest_unix: int,
    cutoff_recent_unix: int,
    older_buckets,
    recent_points,
):
    """
    Apply new 5m snapshots onto existing older_buckets/recent_points.
    Downloads are done concurrently, then snapshots are applied in key order
    to preserve “last value wins” semantics per bucket.
    """
    if not keys:
        return older_buckets, recent_points

    print(f"Processing {len(keys)} new 5m snapshots with up to {SNAPSHOT_FETCH_WORKERS} workers.")

    with ThreadPoolExecutor(max_workers=SNAPSHOT_FETCH_WORKERS) as pool:
        future_map = {key: pool.submit(fetch_snapshot, s3, bucket, key) for key in keys}
        processed = 0
        for key in keys:  # preserve chronological order
            five = future_map[key].result()
            ts_unix = int(five.get("timestamp") or 0)
            if ts_unix <= 0 or ts_unix < cutoff_oldest_unix:
                continue

            data = five.get("data") or {}
            if ts_unix >= cutoff_recent_unix:
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
                bucket_sec = (ts_unix // (OLDER_INTERVAL_MIN * 60)) * (OLDER_INTERVAL_MIN * 60)
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
                    older_buckets[item_id][bucket_sec] = float(mid)

            processed += 1
            if processed % 200 == 0:
                print(f"... processed {processed} new snapshots")

    return older_buckets, recent_points


def write_histories(
    s3,
    bucket: str,
    older_buckets,
    recent_points,
    stored_pairs,
    min_allowed_unix: int,
):
    """
    Combine older_buckets & recent_points into full histories
    and write history/{item_id}.json objects to R2.
    Skip rewriting files whose content would be identical.
    """
    all_item_ids = set(older_buckets.keys()) | set(recent_points.keys()) | set(stored_pairs.keys())
    print("Building histories for", len(all_item_ids), "items.")

    written = 0
    skipped = 0

    for item_id in sorted(all_item_ids):
        pts = []

        older = older_buckets.get(item_id)
        if older:
            for ts, price in older.items():
                if ts >= min_allowed_unix:
                    pts.append((ts, price))

        recent = recent_points.get(item_id)
        if recent:
            for ts, price in recent:
                if ts >= min_allowed_unix:
                    pts.append((ts, price))

        if not pts:
            continue

        pts.sort(key=lambda t: t[0])
        dedup = []
        last_ts = None
        for ts, price in pts:
            if last_ts is not None and ts == last_ts:
                dedup[-1] = (ts, price)
            else:
                dedup.append((ts, price))
                last_ts = ts

        new_pairs = dedup

        existing_pairs = stored_pairs.get(item_id, [])
        if len(existing_pairs) == len(new_pairs) and all(
            (a[0] == b[0] and math.isclose(a[1], b[1])) for a, b in zip(existing_pairs, new_pairs)
        ):
            skipped += 1
            continue

        history_entries = []
        for ts, price in new_pairs:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            iso = dt.isoformat().replace("+00:00", "Z")
            history_entries.append(
                {
                    "timestamp_iso": iso,
                    "price": float(price),
                }
            )

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

    print(f"Done. Wrote history for {written} items. Skipped {skipped} unchanged files.")


def main():
    bucket = os.environ["R2_BUCKET"]
    s3 = get_r2_client()

    now = datetime.now(timezone.utc)
    cutoff_oldest = now - timedelta(days=MAX_HISTORY_DAYS)
    cutoff_recent = now - timedelta(hours=RECENT_WINDOW_HOURS)
    cutoff_oldest_unix = int(cutoff_oldest.timestamp())
    cutoff_recent_unix = int(cutoff_recent.timestamp())
    min_allowed_unix = cutoff_oldest_unix

    meta = load_history_meta(s3, bucket)

    all_keys = list_recent_5m_keys(s3, bucket)
    all_keys.sort()
    print("Found", len(all_keys), "5m snapshots in the last", MAX_HISTORY_DAYS, "days.")

    if meta:
        last_key = meta.get("last_processed_key")
    else:
        last_key = None

    new_keys = all_keys
    seed_from_existing = False
    if last_key and last_key in all_keys:
        idx = all_keys.index(last_key)
        if idx + 1 < len(all_keys):
            new_keys = all_keys[idx + 1 :]
        else:
            new_keys = []
        seed_from_existing = True
    else:
        if last_key:
            print("Previous last_processed_key not found; doing full rebuild.")
        seed_from_existing = False

    print(f"Will process {len(new_keys)} new snapshots (from {len(all_keys)} total).")

    if seed_from_existing and not new_keys:
        print("No new snapshots since last_processed_key; skipping rebuild.")
        return

    older_buckets, recent_points, stored_pairs = load_existing_histories(
        s3, bucket, cutoff_oldest_unix, cutoff_recent_unix
    )

    if not seed_from_existing:
        older_buckets = defaultdict(dict)
        recent_points = defaultdict(list)

    older_buckets, recent_points = apply_new_snapshots(
        s3,
        bucket,
        new_keys,
        cutoff_oldest_unix,
        cutoff_recent_unix,
        older_buckets,
        recent_points,
    )

    write_histories(
        s3,
        bucket,
        older_buckets,
        recent_points,
        stored_pairs,
        min_allowed_unix,
    )

    if all_keys:
        new_meta = {
            "last_processed_key": all_keys[-1],
            "generated_at_iso": now.isoformat(),
        }
        save_history_meta(s3, bucket, new_meta)
        print("Updated history meta with last_processed_key:", all_keys[-1])


if __name__ == "__main__":
    main()
