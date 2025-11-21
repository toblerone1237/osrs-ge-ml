import os
import json
from datetime import datetime, timezone

import boto3
from botocore.config import Config
import pandas as pd
import numpy as np

# --- Multi-horizon configuration -------------------------------------------

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
    Load a single 5-minute snapshot JSON from R2.
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    snap = json.loads(obj["Body"].read())
    return snap


def flatten_5m_snapshots(s3, bucket, keys):
    """
    Flatten a list of OSRS 5-minute snapshot objects into a tabular DataFrame.

    Columns:
      - timestamp: timezone-aware datetime (UTC) of the 5m bucket
      - timestamp_unix: integer seconds since epoch (UTC)
      - item_id
      - avg_high_price, avg_low_price
      - high_volume, low_volume
    """
    rows = []
    for key in keys:
        snap = load_5m_snapshot(s3, bucket, key)
        five = snap["five_minute"]
        ts_unix = five["timestamp"]  # seconds since epoch (int)
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


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic per-snapshot features used by other helpers.

    Adds:
      - mid_price
      - spread, spread_pct
      - total_volume_5m
      - log_volume_5m
    """
    df = df.copy()
    df["mid_price"] = (df["avg_high_price"] + df["avg_low_price"]) / 2.0

    df["spread"] = df["avg_high_price"] - df["avg_low_price"]
    # avoid division by zero
    df["spread_pct"] = df["spread"] / df["mid_price"].replace(0, np.nan)

    df["total_volume_5m"] = (df["high_volume"] + df["low_volume"]).fillna(0)
    df["log_volume_5m"] = np.log1p(df["total_volume_5m"])

    return df


def compute_return_scale(
    df: pd.DataFrame,
    tick_size: float = 1.0,
    min_scale: float = 1e-3,
    max_scale: float = 5.0,
    volatility_col: str = "volatility_60m",
    robust_vol_col: str = "volatility_60m_mad",
    spread_col: str = "spread_pct",
    mid_col: str = "mid_price",
) -> pd.Series:
    """
    Build a per-row scaling factor for returns using recent volatility plus a tick floor.

    This is used to train/predict in volatility units so microcap buckets (where 1 tick
    is a huge %) cannot dominate the loss.
    """
    mid = df[mid_col].to_numpy(dtype="float64")
    mid_safe = np.where(np.isfinite(mid) & (mid > 0), mid, tick_size)

    vol = df.get(volatility_col)
    if vol is None:
        vol_arr = np.zeros_like(mid_safe)
    else:
        vol_arr = vol.to_numpy(dtype="float64")
    vol_arr = np.where(np.isfinite(vol_arr), vol_arr, 0.0)

    robust_vol = df.get(robust_vol_col)
    if robust_vol is None:
        robust_vol_arr = np.zeros_like(mid_safe)
    else:
        robust_vol_arr = robust_vol.to_numpy(dtype="float64")
    robust_vol_arr = np.where(np.isfinite(robust_vol_arr), robust_vol_arr, 0.0)

    spread = df.get(spread_col)
    if spread is None:
        spread_arr = np.zeros_like(mid_safe)
    else:
        spread_arr = spread.to_numpy(dtype="float64")
    spread_arr = np.where(np.isfinite(spread_arr), np.abs(spread_arr), 0.0)

    tick_pct = tick_size / np.maximum(mid_safe, tick_size)
    vol_combined = np.maximum(vol_arr, robust_vol_arr)
    scale = np.maximum(vol_combined, tick_pct)
    scale = np.maximum(scale, spread_arr)
    scale = np.clip(scale, min_scale, max_scale)

    return pd.Series(scale, index=df.index, name="return_scale")


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all features used by the ML models (training & scoring).

    This builds on add_basic_features and then adds:
      - Price/volume basics: mid_price, spread (+ pct), raw high/low prices
      - Lagged returns: 5m/10m/15m/30m/60m
      - Volatility: 30m + 60m (std and MAD)
      - Rolling volume + ratios: 30m/60m sums, z-score, imbalance
      - Spread posture: spread_pct vs 60m median
      - Price posture: z-score and distance to 60m high/low
      - Time encodings: hour/dow sin-cos
    """
    df = add_basic_features(df)
    df = df.dropna(subset=["mid_price"]).copy()
    df = df[df["mid_price"] > 0].copy()
    df.sort_values(["item_id", "timestamp"], inplace=True)

    # Group by item for lag/rolling features
    grp_mid = df.groupby("item_id", sort=False)["mid_price"]
    grp_vol = df.groupby("item_id", sort=False)["total_volume_5m"]
    grp_spread_pct = df.groupby("item_id", sort=False)["spread_pct"]

    # Past returns (assuming 5m spacing between rows within an item series)
    df["ret_5m_past"] = grp_mid.pct_change(periods=1)
    df["ret_10m_past"] = grp_mid.pct_change(periods=2)
    df["ret_15m_past"] = grp_mid.pct_change(periods=3)
    df["ret_30m_past"] = grp_mid.pct_change(periods=6)
    df["ret_60m_past"] = grp_mid.pct_change(periods=12)

    # Volatility of 5m returns on short/long windows
    grp_ret5 = df.groupby("item_id", sort=False)["ret_5m_past"]
    roll_ret_30m = grp_ret5.rolling(window=6, min_periods=2)
    roll_ret_60m = grp_ret5.rolling(window=12, min_periods=2)

    df["volatility_30m"] = (
        roll_ret_30m.std().reset_index(level=0, drop=True)
    )
    mad_rolling_30 = (
        roll_ret_30m.apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=False
        ).reset_index(level=0, drop=True)
    )
    df["volatility_30m_mad"] = 1.4826 * mad_rolling_30

    df["volatility_60m"] = (
        roll_ret_60m.std().reset_index(level=0, drop=True)
    )
    # Robust (MAD-based) 60m volatility to handle flat/zero-return windows
    mad_rolling = (
        roll_ret_60m.apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=False
        ).reset_index(level=0, drop=True)
    )
    df["volatility_60m_mad"] = 1.4826 * mad_rolling

    # Rolling volume windows (30m/60m)
    rolling_vol_30m = (
        grp_vol.rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rolling_vol_60m = (
        grp_vol.rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["rolling_volume_30m"] = rolling_vol_30m
    df["log_rolling_volume_30m"] = np.log1p(rolling_vol_30m)

    df["rolling_volume_60m"] = rolling_vol_60m
    df["log_rolling_volume_60m"] = np.log1p(rolling_vol_60m)

    # Relative volume: current 5m vs average over last 60m
    avg_vol_30m = rolling_vol_30m / 6.0
    avg_vol_60m = rolling_vol_60m / 12.0
    df["volume_ratio_5m_to_30m"] = df["total_volume_5m"] / (1.0 + avg_vol_30m)
    df["volume_ratio_5m_to_60m"] = df["total_volume_5m"] / (1.0 + avg_vol_60m)

    # Volume z-score (vs 60m mean/std)
    rolling_vol_mean_60m = (
        grp_vol.rolling(window=12, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_vol_std_60m = (
        grp_vol.rolling(window=12, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["volume_zscore_60m"] = (df["total_volume_5m"] - rolling_vol_mean_60m) / (
        1.0 + rolling_vol_std_60m
    )

    # Volume imbalance (side vols) and rolling imbalance posture
    df["volume_imbalance"] = (
        df["high_volume"].fillna(0.0) - df["low_volume"].fillna(0.0)
    )
    df["volume_imbalance_ratio"] = df["volume_imbalance"] / (
        df["total_volume_5m"] + 1.0
    )
    grp_imb = df.groupby("item_id", sort=False)["volume_imbalance"]
    rolling_imbalance_60m = (
        grp_imb.rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["rolling_volume_imbalance_60m"] = rolling_imbalance_60m
    df["volume_imbalance_ratio_60m"] = rolling_imbalance_60m / (
        1.0 + rolling_vol_60m
    )

    # Spread posture vs recent median
    spread_pct_median_60m = (
        grp_spread_pct.rolling(window=12, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    df["spread_pct_median_60m"] = spread_pct_median_60m
    df["spread_pct_delta_from_median_60m"] = (
        df["spread_pct"] - spread_pct_median_60m
    )

    # Price posture vs trailing distribution
    rolling_mid_mean_60m = (
        grp_mid.rolling(window=12, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_mid_std_60m = (
        grp_mid.rolling(window=12, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["price_zscore_60m"] = (df["mid_price"] - rolling_mid_mean_60m) / (
        1e-9 + rolling_mid_std_60m
    )

    rolling_mid_max_60m = (
        grp_mid.rolling(window=12, min_periods=2)
        .max()
        .reset_index(level=0, drop=True)
    )
    rolling_mid_min_60m = (
        grp_mid.rolling(window=12, min_periods=2)
        .min()
        .reset_index(level=0, drop=True)
    )
    max_safe = rolling_mid_max_60m.replace(0, np.nan)
    min_safe = rolling_mid_min_60m.replace(0, np.nan)
    df["pct_from_60m_high"] = df["mid_price"] / max_safe - 1.0
    df["pct_from_60m_low"] = df["mid_price"] / min_safe - 1.0

    # Time-of-day / day-of-week encodings
    ts = df["timestamp"]
    # ts should already be tz-aware UTC, but be safe:
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone.utc)
    else:
        ts = ts.dt.tz_convert(timezone.utc)

    hour_float = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek  # 0=Monday

    df["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    # Clean up NaNs / infs in derived features (leave mid_price, etc. as-is)
    derived_cols = [
        "ret_5m_past",
        "ret_10m_past",
        "ret_15m_past",
        "ret_30m_past",
        "ret_60m_past",
        "volatility_30m",
        "volatility_30m_mad",
        "volatility_60m",
        "volatility_60m_mad",
        "rolling_volume_30m",
        "log_rolling_volume_30m",
        "rolling_volume_60m",
        "log_rolling_volume_60m",
        "volume_ratio_5m_to_30m",
        "volume_ratio_5m_to_60m",
        "volume_zscore_60m",
        "volume_imbalance",
        "volume_imbalance_ratio",
        "rolling_volume_imbalance_60m",
        "volume_imbalance_ratio_60m",
        "spread_pct_median_60m",
        "spread_pct_delta_from_median_60m",
        "price_zscore_60m",
        "pct_from_60m_high",
        "pct_from_60m_low",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    df[derived_cols] = df[derived_cols].replace([np.inf, -np.inf], np.nan)
    df[derived_cols] = df[derived_cols].fillna(0.0)

    return df


def add_multi_horizon_returns(
    df: pd.DataFrame,
    horizons_minutes=None,
    item_col: str = "item_id",
    time_col: str = "timestamp_unix",
    mid_col: str = "mid_price",
) -> pd.DataFrame:
    """
    For each row (item_id, timestamp), compute future returns at multiple horizons.

    For each horizon h in `horizons_minutes` (minutes), we compute:

        ret_hm = (mid(t + h) / mid(t)) - 1

    where mid(t+h) is the mid_price at the first snapshot whose timestamp
    is >= t + h minutes (within the same item_id series).

    This function assumes:
      - df[time_col] is integer seconds since epoch (e.g. timestamp_unix)
      - df[mid_col] holds the mid price at that time
      - df is already at 5-minute resolution per item_id (your 5m snapshots)

    It returns a *copy* of df with new columns:

        ret_5m, ret_10m, ..., ret_120m

    (or whatever horizons you pass in).
    """
    if horizons_minutes is None:
        horizons_minutes = HORIZONS_MINUTES

    df = df.copy()
    df.sort_values([item_col, time_col], inplace=True)

    # Prepare result columns
    for h in horizons_minutes:
        df[f"ret_{h}m"] = np.nan

    # Group per item to keep lookup local
    for _, g in df.groupby(item_col, sort=False):
        idx = g.index.to_numpy()
        ts = g[time_col].to_numpy(dtype="int64")  # seconds
        mid = g[mid_col].to_numpy(dtype="float64")

        n = len(ts)
        if n == 0:
            continue

        valid_mid = np.isfinite(mid) & (mid > 0)
        if not valid_mid.any():
            continue

        for h in horizons_minutes:
            horizon_sec = int(h) * 60
            targets = ts + horizon_sec
            # For each current timestamp, find index of first ts[j] >= target
            j = np.searchsorted(ts, targets, side="left")

            mask = (j < n) & valid_mid
            if not np.any(mask):
                continue

            future_mid = np.full_like(mid, np.nan, dtype="float64")
            future_mid[mask] = mid[j[mask]]
            mask &= np.isfinite(future_mid) & (future_mid > 0)

            if not np.any(mask):
                continue

            ret = np.full_like(mid, np.nan, dtype="float64")
            ret[mask] = (future_mid[mask] / mid[mask]) - 1.0

            df.loc[idx, f"ret_{h}m"] = ret

    return df
