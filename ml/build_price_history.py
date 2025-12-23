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
HISTORY_FETCH_WORKERS = int(os.environ.get("HISTORY_FETCH_WORKERS", "16"))
HISTORY_WRITE_WORKERS = int(os.environ.get("HISTORY_WRITE_WORKERS", "16"))

# History file schema version:
# - v1: {"timestamp_iso","price"}
# - v2: + {"volume"}
# - v3: + {"avg_high_price","avg_low_price","high_volume","low_volume"}
HISTORY_SCHEMA_VERSION = 3


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
    # item_id -> list[(ts_unix, price, volume, avg_high_price, avg_low_price, high_volume, low_volume)]
    # for histories that match HISTORY_SCHEMA_VERSION, or None for legacy histories (forces a rewrite).
    stored_pairs = {}

    history_keys = list_keys_with_prefix(s3, bucket, "history/")
    history_keys = [k for k in history_keys if k.endswith(".json") and k != HISTORY_META_KEY]
    history_keys.sort()

    def parse_pos(v):
        if v is None:
            return None
        try:
            f = float(v)
        except Exception:
            return None
        if not math.isfinite(f) or f <= 0:
            return None
        return float(f)

    def parse_nonneg(v):
        if v is None:
            return None
        try:
            f = float(v)
        except Exception:
            return None
        if not math.isfinite(f) or f < 0:
            return None
        return float(f)

    def sum_or_none(a, b):
        if a is None or b is None:
            return None
        return float(a + b)

    def load_one(key):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return key, json.loads(obj["Body"].read())
        except Exception:
            return key, None

    with ThreadPoolExecutor(max_workers=HISTORY_FETCH_WORKERS) as pool:
        futures = [pool.submit(load_one, key) for key in history_keys]
        for idx, fut in enumerate(as_completed(futures), start=1):
            key, data = fut.result()
            if data is None:
                continue
            try:
                file_schema_version = int(data.get("schema_version")) if data.get("schema_version") is not None else None
            except Exception:
                file_schema_version = None
            needs_schema_upgrade = file_schema_version != HISTORY_SCHEMA_VERSION

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

                volume_raw = entry.get("volume")
                volume_f = parse_nonneg(volume_raw) if volume_raw is not None else None

                ah_raw = entry.get("avg_high_price")
                if ah_raw is None and "avgHighPrice" in entry:
                    ah_raw = entry.get("avgHighPrice")
                al_raw = entry.get("avg_low_price")
                if al_raw is None and "avgLowPrice" in entry:
                    al_raw = entry.get("avgLowPrice")
                avg_high_f = parse_pos(ah_raw)
                avg_low_f = parse_pos(al_raw)

                hv_raw = entry.get("high_volume")
                if hv_raw is None and "highPriceVolume" in entry:
                    hv_raw = entry.get("highPriceVolume")
                lv_raw = entry.get("low_volume")
                if lv_raw is None and "lowPriceVolume" in entry:
                    lv_raw = entry.get("lowPriceVolume")
                high_vol_f = parse_nonneg(hv_raw)
                low_vol_f = parse_nonneg(lv_raw)

                if volume_f is None:
                    if high_vol_f is not None or low_vol_f is not None:
                        volume_f = float((high_vol_f or 0.0) + (low_vol_f or 0.0))
                    else:
                        volume_f = 0.0

                ts_unix = parse_iso_to_unix(str(ts_iso))
                if ts_unix is None:
                    continue
                pairs.append((ts_unix, price_f, volume_f, avg_high_f, avg_low_f, high_vol_f, low_vol_f))

                if ts_unix < cutoff_oldest_unix:
                    continue
                if ts_unix >= cutoff_recent_unix:
                    recent_points[item_id].append(
                        (ts_unix, price_f, volume_f, avg_high_f, avg_low_f, high_vol_f, low_vol_f)
                    )
                else:
                    # For existing histories we treat the stored timestamp as canonical.
                    prev = older_buckets[item_id].get(ts_unix)
                    if prev is None:
                        older_buckets[item_id][ts_unix] = (
                            price_f,
                            volume_f,
                            avg_high_f,
                            avg_low_f,
                            high_vol_f,
                            low_vol_f,
                        )
                    else:
                        prev_price, prev_volume, prev_ah, prev_al, prev_hv, prev_lv = prev

                        older_buckets[item_id][ts_unix] = (
                            price_f,
                            float(prev_volume + volume_f),
                            avg_high_f,
                            avg_low_f,
                            sum_or_none(prev_hv, high_vol_f),
                            sum_or_none(prev_lv, low_vol_f),
                        )

            pairs.sort(key=lambda t: t[0])
            dedup_points = []
            last_ts = None
            for ts, price, volume, avg_high, avg_low, high_vol, low_vol in pairs:
                if last_ts is not None and ts == last_ts:
                    # Last values win for prices; volumes aggregate.
                    (
                        prev_ts,
                        prev_price,
                        prev_volume,
                        prev_avg_high,
                        prev_avg_low,
                        prev_high_vol,
                        prev_low_vol,
                    ) = dedup_points[-1]

                    def sum_or_none(a, b):
                        if a is None or b is None:
                            return None
                        return float(a + b)

                    dedup_points[-1] = (
                        ts,
                        price,
                        float(prev_volume + volume),
                        avg_high,
                        avg_low,
                        sum_or_none(prev_high_vol, high_vol),
                        sum_or_none(prev_low_vol, low_vol),
                    )
                else:
                    dedup_points.append((ts, price, volume, avg_high, avg_low, high_vol, low_vol))
                    last_ts = ts

            if needs_schema_upgrade:
                stored_pairs[item_id] = None
            else:
                stored_pairs[item_id] = dedup_points

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

    def parse_price(v):
        if v is None:
            return None
        try:
            f = float(v)
        except Exception:
            return None
        if not math.isfinite(f) or f <= 0:
            return None
        return f

    def parse_volume(v):
        if v is None:
            return 0.0
        try:
            f = float(v)
        except Exception:
            return 0.0
        if not math.isfinite(f) or f < 0:
            return 0.0
        return f

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
                    high_vol_f = parse_volume(st.get("highPriceVolume"))
                    low_vol_f = parse_volume(st.get("lowPriceVolume"))
                    total_vol = float(high_vol_f + low_vol_f)

                    ah_f = parse_price(st.get("avgHighPrice"))
                    al_f = parse_price(st.get("avgLowPrice"))
                    if ah_f is None and al_f is None:
                        continue
                    if ah_f is not None and al_f is not None:
                        if high_vol_f <= 0 and low_vol_f > 0:
                            mid = al_f
                        elif low_vol_f <= 0 and high_vol_f > 0:
                            mid = ah_f
                        else:
                            mid = (ah_f + al_f) / 2.0
                    else:
                        mid = ah_f if ah_f is not None else al_f
                    if mid is None or mid <= 0 or not math.isfinite(mid):
                        continue
                    try:
                        item_id = int(item_id_str)
                    except ValueError:
                        continue
                    recent_points[item_id].append(
                        (
                            ts_unix,
                            float(mid),
                            total_vol,
                            float(ah_f) if ah_f is not None else None,
                            float(al_f) if al_f is not None else None,
                            float(high_vol_f),
                            float(low_vol_f),
                        )
                    )
            else:
                bucket_sec = (ts_unix // (OLDER_INTERVAL_MIN * 60)) * (OLDER_INTERVAL_MIN * 60)
                for item_id_str, st in data.items():
                    high_vol_f = parse_volume(st.get("highPriceVolume"))
                    low_vol_f = parse_volume(st.get("lowPriceVolume"))
                    total_vol = float(high_vol_f + low_vol_f)

                    ah_f = parse_price(st.get("avgHighPrice"))
                    al_f = parse_price(st.get("avgLowPrice"))
                    if ah_f is None and al_f is None:
                        continue
                    if ah_f is not None and al_f is not None:
                        if high_vol_f <= 0 and low_vol_f > 0:
                            mid = al_f
                        elif low_vol_f <= 0 and high_vol_f > 0:
                            mid = ah_f
                        else:
                            mid = (ah_f + al_f) / 2.0
                    else:
                        mid = ah_f if ah_f is not None else al_f
                    if mid is None or mid <= 0 or not math.isfinite(mid):
                        continue
                    try:
                        item_id = int(item_id_str)
                    except ValueError:
                        continue
                    prev = older_buckets[item_id].get(bucket_sec)
                    if prev is None:
                        older_buckets[item_id][bucket_sec] = (
                            float(mid),
                            total_vol,
                            float(ah_f) if ah_f is not None else None,
                            float(al_f) if al_f is not None else None,
                            float(high_vol_f),
                            float(low_vol_f),
                        )
                    else:
                        prev_price, prev_vol, prev_ah, prev_al, prev_high_vol, prev_low_vol = prev

                        def add_vol_or_none(prev_v, add_v):
                            if prev_v is None:
                                return None
                            return float(prev_v + add_v)

                        # Last price wins; volumes accumulate within the bucket.
                        older_buckets[item_id][bucket_sec] = (
                            float(mid),
                            float(prev_vol + total_vol),
                            float(ah_f) if ah_f is not None else None,
                            float(al_f) if al_f is not None else None,
                            add_vol_or_none(prev_high_vol, float(high_vol_f)),
                            add_vol_or_none(prev_low_vol, float(low_vol_f)),
                        )

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

    to_write = []
    skipped = 0

    for item_id in sorted(all_item_ids):
        pts = []

        older = older_buckets.get(item_id)
        if older:
            for ts, pv in older.items():
                price, volume, avg_high, avg_low, high_vol, low_vol = pv
                if ts >= min_allowed_unix:
                    pts.append((ts, price, volume, avg_high, avg_low, high_vol, low_vol))

        recent = recent_points.get(item_id)
        if recent:
            for ts, price, volume, avg_high, avg_low, high_vol, low_vol in recent:
                if ts >= min_allowed_unix:
                    pts.append((ts, price, volume, avg_high, avg_low, high_vol, low_vol))

        if not pts:
            continue

        pts.sort(key=lambda t: t[0])
        dedup = []
        last_ts = None
        for ts, price, volume, avg_high, avg_low, high_vol, low_vol in pts:
            if last_ts is not None and ts == last_ts:
                (
                    prev_ts,
                    prev_price,
                    prev_volume,
                    prev_avg_high,
                    prev_avg_low,
                    prev_high_vol,
                    prev_low_vol,
                ) = dedup[-1]

                def sum_or_none(a, b):
                    if a is None or b is None:
                        return None
                    return float(a + b)

                dedup[-1] = (
                    ts,
                    price,
                    float(prev_volume + volume),
                    avg_high,
                    avg_low,
                    sum_or_none(prev_high_vol, high_vol),
                    sum_or_none(prev_low_vol, low_vol),
                )
            else:
                dedup.append((ts, price, volume, avg_high, avg_low, high_vol, low_vol))
                last_ts = ts

        def same_opt_num(a, b):
            if a is None or b is None:
                return a is None and b is None
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)

        existing_points = stored_pairs.get(item_id)
        if existing_points is not None and len(existing_points) == len(dedup) and all(
            (
                a[0] == b[0]
                and same_opt_num(a[1], b[1])
                and same_opt_num(a[2], b[2])
                and same_opt_num(a[3], b[3])
                and same_opt_num(a[4], b[4])
                and same_opt_num(a[5], b[5])
                and same_opt_num(a[6], b[6])
            )
            for a, b in zip(existing_points, dedup)
        ):
            skipped += 1
            continue

        to_write.append((item_id, dedup))

    written = 0
    if to_write:
        def write_one(item_id, points):
            history_entries = []
            for ts, price, volume, avg_high, avg_low, high_vol, low_vol in points:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                iso = dt.isoformat().replace("+00:00", "Z")
                history_entries.append(
                    {
                        "timestamp_iso": iso,
                        "price": float(price),
                        "volume": float(volume),
                        "avg_high_price": float(avg_high) if avg_high is not None else None,
                        "avg_low_price": float(avg_low) if avg_low is not None else None,
                        "high_volume": float(high_vol) if high_vol is not None else None,
                        "low_volume": float(low_vol) if low_vol is not None else None,
                    }
                )

            out = {
                "schema_version": HISTORY_SCHEMA_VERSION,
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
            return item_id

        with ThreadPoolExecutor(max_workers=HISTORY_WRITE_WORKERS) as pool:
            futures = [pool.submit(write_one, item_id, triplets) for item_id, triplets in to_write]
            for fut in as_completed(futures):
                fut.result()
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

    force_full_rebuild = os.environ.get("FORCE_FULL_REBUILD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if force_full_rebuild:
        print("FORCE_FULL_REBUILD enabled; doing full rebuild from raw 5m snapshots.")

    last_key = meta.get("last_processed_key") if meta and not force_full_rebuild else None

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

    meta_schema_version = None
    if isinstance(meta, dict):
        try:
            meta_schema_version = (
                int(meta.get("schema_version")) if meta.get("schema_version") is not None else None
            )
        except Exception:
            meta_schema_version = None
    needs_schema_migration = meta_schema_version != HISTORY_SCHEMA_VERSION
    if seed_from_existing and not new_keys and not needs_schema_migration:
        print("No new snapshots since last_processed_key and schema is up-to-date; skipping rebuild.")
        return
    if seed_from_existing and not new_keys and needs_schema_migration:
        print("No new snapshots, but history schema changed; migrating existing history files.")

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
            "schema_version": HISTORY_SCHEMA_VERSION,
        }
        save_history_meta(s3, bucket, new_meta)
        print("Updated history meta with last_processed_key:", all_keys[-1])


if __name__ == "__main__":
    main()
