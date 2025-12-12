import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from math import exp, sqrt, pi
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from features import get_r2_client, list_keys_with_prefix


WINDOW_DAYS = int(os.getenv("PEAKS_WINDOW_DAYS", "14"))
MAX_ITER = int(os.getenv("PEAKS_EM_MAX_ITER", "15"))
FETCH_WORKERS = int(os.getenv("PEAKS_FETCH_WORKERS", "16"))


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    s = max(float(sigma), 1e-6)
    z = (x - mu) / s
    return np.exp(-0.5 * z * z) / (s * sqrt(2 * pi))


def fit_two_gaussian_mixture_log(prices: np.ndarray, max_iter: int = MAX_ITER):
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 10:
        return None

    xs = np.log(prices.astype("float64"))
    xs = xs[np.isfinite(xs)]
    if xs.size < 10:
        return None

    xs_sorted = np.sort(xs)
    n = xs.size

    mu1 = float(np.quantile(xs_sorted, 0.25))
    mu2 = float(np.quantile(xs_sorted, 0.90))
    global_mu = float(xs_sorted.mean())
    global_sigma = float(xs_sorted.std())
    global_sigma = max(global_sigma, 1e-4)

    sigma1 = global_sigma
    sigma2 = global_sigma * 2.0
    w1, w2 = 0.9, 0.1

    gamma2 = np.zeros(n, dtype="float64")

    for _ in range(max_iter):
        p1 = w1 * gaussian_pdf(xs, mu1, sigma1)
        p2 = w2 * gaussian_pdf(xs, mu2, sigma2)
        denom = p1 + p2
        gamma2 = np.where(denom > 0, p2 / denom, 0.0)
        g1 = 1.0 - gamma2

        sum1 = float(g1.sum())
        sum2 = float(gamma2.sum())
        if sum1 < 1e-6 or sum2 < 1e-6:
            break

        mu1_new = float((g1 * xs).sum() / sum1)
        mu2_new = float((gamma2 * xs).sum() / sum2)

        var1 = float((g1 * (xs - mu1_new) ** 2).sum() / sum1)
        var2 = float((gamma2 * (xs - mu2_new) ** 2).sum() / sum2)
        sigma1_new = max(sqrt(max(var1, 1e-8)), 1e-4)
        sigma2_new = max(sqrt(max(var2, 1e-8)), 1e-4)

        w1_new = sum1 / n
        w2_new = sum2 / n

        delta = abs(mu1_new - mu1) + abs(mu2_new - mu2)
        mu1, mu2 = mu1_new, mu2_new
        sigma1, sigma2 = sigma1_new, sigma2_new
        w1, w2 = w1_new, w2_new

        if delta < 1e-4:
            break

    # Ensure baseline (mu1) < spike (mu2)
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1
        w1, w2 = w2, w1
        gamma2 = 1.0 - gamma2

    return {
        "mu1": mu1,
        "mu2": mu2,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "w1": w1,
        "w2": w2,
        "gamma2": gamma2,
    }


def compute_catching_peaks_metric(
    history: List[Dict[str, Any]],
    window_days: int = WINDOW_DAYS,
) -> Optional[Dict[str, float]]:
    if not history or len(history) < 30:
        return None

    now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
    cutoff_ms = now_ms - window_days * 86400 * 1000.0
    cutoff24h_ms = now_ms - 24 * 3600 * 1000.0

    pts: List[Tuple[float, float, float]] = []
    for pt in history:
        iso = pt.get("timestamp_iso") or pt.get("timestamp")
        if not iso:
            continue
        try:
            ts = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000.0
        except Exception:
            continue
        if ts < cutoff_ms:
            continue
        price = pt.get("price")
        if price is None:
            continue
        try:
            price_f = float(price)
        except Exception:
            continue
        if not np.isfinite(price_f) or price_f <= 0:
            continue
        vol = pt.get("volume", 0.0)
        try:
            vol_f = float(vol) if vol is not None else 0.0
        except Exception:
            vol_f = 0.0
        if not np.isfinite(vol_f) or vol_f < 0:
            vol_f = 0.0
        pts.append((ts, price_f, vol_f))

    if len(pts) < 30:
        return None

    pts.sort(key=lambda x: x[0])
    prices = np.array([p for _, p, _ in pts], dtype="float64")

    fit = fit_two_gaussian_mixture_log(prices, max_iter=MAX_ITER)
    if fit is None:
        return None

    gamma2 = fit["gamma2"]
    is_spike = gamma2 > 0.5

    baseline_prices = prices[~is_spike]
    spike_prices = prices[is_spike]

    low_avg = float(baseline_prices.mean()) if baseline_prices.size else exp(fit["mu1"])
    peak_avg = float(spike_prices.mean()) if spike_prices.size else exp(fit["mu2"])

    if not np.isfinite(low_avg) or low_avg <= 0 or not np.isfinite(peak_avg):
        return None

    separation = peak_avg / low_avg
    pct_diff = (separation - 1.0) * 100.0
    w2 = float(gamma2.mean())

    baseline_cv = 1.0
    if baseline_prices.size >= 3:
        b_std = float(baseline_prices.std())
        if np.isfinite(b_std) and low_avg > 0:
            baseline_cv = b_std / low_avg

    # Average length of spike runs (posterior > 0.5)
    spike_runs: List[int] = []
    current_run = 0
    for flag in is_spike.tolist():
        if flag:
            current_run += 1
        else:
            if current_run:
                spike_runs.append(current_run)
                current_run = 0
    if current_run:
        spike_runs.append(current_run)

    avg_spike_len = float(np.mean(spike_runs)) if spike_runs else 0.0
    peaks_count = int(len(spike_runs))

    baseline_stability = 1.0 / (1.0 + baseline_cv * 5.0)
    rare_weight = exp(-w2 * 8.0)
    spike_shortness = 1.0 / (1.0 + avg_spike_len / 3.0)
    separation_delta = max(0.0, separation - 1.0)

    score = separation_delta * rare_weight * baseline_stability * spike_shortness
    if not np.isfinite(score):
        score = 0.0

    volume24h = 0.0
    for ts, _, vol in pts:
        if ts >= cutoff24h_ms:
            volume24h += vol

    return {
        "low_avg_price": low_avg,
        "peak_avg_price": peak_avg,
        "pct_difference": pct_diff,
        "volume_24h": volume24h,
        "peaks_count": peaks_count,
        "score": float(score),
    }


def load_latest_mapping(s3, bucket) -> Dict[int, str]:
    now = datetime.now(timezone.utc)
    for delta in range(7):
        d = now - timedelta(days=delta)
        prefix = f"daily/{d.year}/{d.month:02d}/{d.day:02d}"
        keys = list_keys_with_prefix(s3, bucket, prefix)
        if not keys:
            continue
        keys.sort()
        latest_key = keys[-1]
        obj = s3.get_object(Bucket=bucket, Key=latest_key)
        data = json.loads(obj["Body"].read())
        mapping = data.get("mapping")
        if isinstance(mapping, list):
            out = {}
            for m in mapping:
                try:
                    item_id = int(m.get("id"))
                    name = m.get("name")
                except Exception:
                    continue
                if item_id and isinstance(name, str) and name:
                    out[item_id] = name
            if out:
                print("Loaded mapping from", latest_key, "(", len(out), "items )")
            return out
    return {}


def main():
    s3 = get_r2_client()
    bucket = os.environ["R2_BUCKET"]

    mapping = load_latest_mapping(s3, bucket)

    keys = list_keys_with_prefix(s3, bucket, "history/")
    keys = [
        k
        for k in keys
        if k.endswith(".json") and not k.endswith("_meta.json") and "/_" not in k
    ]
    keys.sort()
    print("Found", len(keys), "history files.")

    results: List[Dict[str, Any]] = []

    def process_key(key: str):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            history = data.get("history") or []
            metric = compute_catching_peaks_metric(history, window_days=WINDOW_DAYS)
            if metric is None:
                return None

            base = key.rsplit("/", 1)[-1].split(".", 1)[0]
            item_id = int(base)
            name = mapping.get(item_id) or data.get("name") or f"Item {item_id}"

            metric["item_id"] = item_id
            metric["name"] = name
            return metric
        except Exception as e:
            print("Failed to process", key, ":", e)
            return None

    with ThreadPoolExecutor(max_workers=max(1, FETCH_WORKERS)) as ex:
        futures = [ex.submit(process_key, k) for k in keys]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    print("Computed metrics for", len(results), "items.")

    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y/%m/%d")

    out = {
        "generated_at_iso": now.isoformat(),
        "window_days": WINDOW_DAYS,
        "items_scanned": len(keys),
        "items": results,
    }

    body = json.dumps(out).encode("utf-8")
    key_dated = f"signals/peaks/{date_part}.json"
    key_latest = "signals/peaks/latest.json"

    s3.put_object(Bucket=bucket, Key=key_dated, Body=body)
    s3.put_object(Bucket=bucket, Key=key_latest, Body=body)

    print("Wrote catching-peaks signals to", key_dated, "and updated latest.")


if __name__ == "__main__":
    main()
