import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from math import exp, sqrt, pi
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from features import get_r2_client, list_keys_with_prefix


WINDOW_DAYS = int(os.getenv("PEAKS_WINDOW_DAYS", "14"))
BASELINE_HALF_WINDOW_DAYS = int(os.getenv("PEAKS_BASELINE_HALF_WINDOW_DAYS", "3"))
MAX_ITER = int(os.getenv("PEAKS_EM_MAX_ITER", "15"))
FETCH_WORKERS = int(os.getenv("PEAKS_FETCH_WORKERS", "16"))
TAX_RATE = float(os.getenv("PEAKS_TAX_RATE", "0.02"))
FLIP_BUY_Q = float(os.getenv("PEAKS_FLIP_BUY_Q", "0.10"))
FLIP_SELL_Q = float(os.getenv("PEAKS_FLIP_SELL_Q", "0.90"))


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
    trade_limit_4h: Optional[float] = None,
    tax_rate: float = TAX_RATE,
) -> Optional[Dict[str, Any]]:
    if not history or len(history) < 30:
        return None

    now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
    cutoff_ms = now_ms - window_days * 86400 * 1000.0
    cutoff24h_ms = now_ms - 24 * 3600 * 1000.0

    def parse_optional_float(v) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
        except Exception:
            return None
        if not np.isfinite(f):
            return None
        return float(f)

    def parse_volume(v) -> float:
        if v is None:
            return 0.0
        try:
            f = float(v)
        except Exception:
            return 0.0
        if not np.isfinite(f) or f < 0:
            return 0.0
        return float(f)

    pts: List[Tuple[float, float, float]] = []
    side_pts: List[Tuple[float, float, float, float, float, float, float]] = []
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
        vol_f = parse_volume(pt.get("volume", 0.0))
        pts.append((ts, price_f, vol_f))

        avg_high = parse_optional_float(pt.get("avg_high_price"))
        avg_low = parse_optional_float(pt.get("avg_low_price"))
        high_vol = parse_volume(pt.get("high_volume"))
        low_vol = parse_volume(pt.get("low_volume"))
        side_pts.append(
            (
                ts,
                price_f,
                vol_f,
                float(avg_high) if avg_high is not None else float("nan"),
                float(avg_low) if avg_low is not None else float("nan"),
                high_vol,
                low_vol,
            )
        )

    if len(pts) < 30:
        return None

    pts.sort(key=lambda x: x[0])
    ts_ms = np.array([ts for ts, _, _ in pts], dtype="float64")
    prices = np.array([p for _, p, _ in pts], dtype="float64")

    # Side-series (if present in history schema v2+): avg_high/avg_low and their volumes.
    side_pts.sort(key=lambda x: x[0])
    avg_high_prices = np.array([ah for _, _, _, ah, _, _, _ in side_pts], dtype="float64")
    avg_low_prices = np.array([al for _, _, _, _, al, _, _ in side_pts], dtype="float64")
    high_volumes = np.array([hv for _, _, _, _, _, hv, _ in side_pts], dtype="float64")
    low_volumes = np.array([lv for _, _, _, _, _, _, lv in side_pts], dtype="float64")

    mean_price = float(np.mean(prices))
    variance = (
        float(np.mean(np.sqrt(np.abs(prices - mean_price))))
        if np.isfinite(mean_price)
        else float("nan")
    )
    variance_pct = (
        float(variance) / float(mean_price) * 100.0
        if np.isfinite(variance) and np.isfinite(mean_price) and mean_price > 0
        else None
    )

    def unweighted_quantile(x: np.ndarray, q: float) -> Optional[float]:
        if not np.isfinite(q):
            return None
        qq = float(max(0.0, min(1.0, q)))
        mask = np.isfinite(x) & (x > 0)
        if not int(mask.sum()):
            return None
        return float(np.quantile(x[mask], qq))

    def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> Optional[float]:
        if not np.isfinite(q):
            return None
        qq = float(max(0.0, min(1.0, q)))
        mask = np.isfinite(x) & (x > 0) & np.isfinite(w) & (w > 0)
        if not int(mask.sum()):
            return None
        xs = x[mask]
        ws = w[mask]
        order = np.argsort(xs)
        xs = xs[order]
        ws = ws[order]
        cum = np.cumsum(ws)
        total = float(cum[-1])
        if not np.isfinite(total) or total <= 0:
            return None
        target = qq * total
        idx = int(np.searchsorted(cum, target, side="left"))
        if idx < 0:
            idx = 0
        if idx >= int(xs.size):
            idx = int(xs.size) - 1
        return float(xs[idx])

    # Flip-oriented metrics: use volume-weighted tails on the buy (avg_low) and sell (avg_high) sides.
    buy_q = float(max(0.0, min(1.0, FLIP_BUY_Q)))
    sell_q = float(max(0.0, min(1.0, FLIP_SELL_Q)))
    flip_buy_price = weighted_quantile(avg_low_prices, low_volumes, buy_q)
    if flip_buy_price is None:
        flip_buy_price = unweighted_quantile(avg_low_prices, buy_q)
    flip_sell_price = weighted_quantile(avg_high_prices, high_volumes, sell_q)
    if flip_sell_price is None:
        flip_sell_price = unweighted_quantile(avg_high_prices, sell_q)

    flip_edge_gp: Optional[float] = None
    flip_edge_pct: Optional[float] = None
    if (
        flip_buy_price is not None
        and flip_sell_price is not None
        and np.isfinite(flip_buy_price)
        and np.isfinite(flip_sell_price)
        and flip_buy_price > 0
    ):
        net_sell = float(flip_sell_price) * (1.0 - float(tax_rate))
        flip_edge_gp = float(net_sell - float(flip_buy_price))
        flip_edge_pct = float(flip_edge_gp / float(flip_buy_price) * 100.0)

    flip_tail_buy_units_total = 0.0
    flip_tail_sell_units_total = 0.0
    if flip_buy_price is not None and np.isfinite(flip_buy_price) and flip_buy_price > 0:
        buy_mask = np.isfinite(avg_low_prices) & (avg_low_prices > 0) & (avg_low_prices <= float(flip_buy_price))
        if int(buy_mask.sum()):
            flip_tail_buy_units_total = float(np.sum(low_volumes[buy_mask]))
    if flip_sell_price is not None and np.isfinite(flip_sell_price) and flip_sell_price > 0:
        sell_mask = np.isfinite(avg_high_prices) & (avg_high_prices > 0) & (avg_high_prices >= float(flip_sell_price))
        if int(sell_mask.sum()):
            flip_tail_sell_units_total = float(np.sum(high_volumes[sell_mask]))

    denom_days = float(window_days) if window_days > 0 else 1.0
    flip_tail_buy_units_per_day = float(flip_tail_buy_units_total / denom_days)
    flip_tail_sell_units_per_day = float(flip_tail_sell_units_total / denom_days)

    flip_flow_balance_pct: Optional[float] = None
    if flip_tail_buy_units_per_day > 0 or flip_tail_sell_units_per_day > 0:
        m = max(flip_tail_buy_units_per_day, flip_tail_sell_units_per_day)
        n = min(flip_tail_buy_units_per_day, flip_tail_sell_units_per_day)
        if m > 0:
            flip_flow_balance_pct = float(n / m * 100.0)

    flip_cap_units_per_day: Optional[float] = None
    if trade_limit_4h is not None:
        try:
            lim = float(trade_limit_4h)
        except Exception:
            lim = float("nan")
        if np.isfinite(lim) and lim > 0:
            flip_cap_units_per_day = float(lim * 6.0)

    flip_units_per_day: Optional[float] = None
    if flip_tail_buy_units_per_day > 0 or flip_tail_sell_units_per_day > 0:
        base_units = min(flip_tail_buy_units_per_day, flip_tail_sell_units_per_day)
        if flip_cap_units_per_day is not None and np.isfinite(flip_cap_units_per_day):
            flip_units_per_day = float(min(base_units, float(flip_cap_units_per_day)))
        else:
            flip_units_per_day = float(base_units)

    flip_expected_gp_per_day: Optional[float] = None
    if flip_edge_gp is not None and flip_units_per_day is not None:
        flip_expected_gp_per_day = float(max(0.0, float(flip_edge_gp)) * float(flip_units_per_day))

    spread_pct_mean: Optional[float] = None
    spread_mask = np.isfinite(avg_high_prices) & (avg_high_prices > 0) & np.isfinite(avg_low_prices) & (avg_low_prices > 0)
    if int(spread_mask.sum()):
        mids = (avg_high_prices[spread_mask] + avg_low_prices[spread_mask]) / 2.0
        valid_mid = np.isfinite(mids) & (mids > 0)
        if int(valid_mid.sum()):
            mids = mids[valid_mid]
            ah = avg_high_prices[spread_mask][valid_mid]
            al = avg_low_prices[spread_mask][valid_mid]
            spread_pct = (ah - al) / mids * 100.0
            weights = high_volumes[spread_mask][valid_mid] + low_volumes[spread_mask][valid_mid]
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
            wsum = float(np.sum(weights))
            if wsum > 0:
                spread_pct_mean = float(np.sum(spread_pct * weights) / wsum)
            else:
                spread_pct_mean = float(np.mean(spread_pct))

    # Cycle stats: how often we see a "tail buy" then later a "tail sell".
    flip_cycles = 0
    flip_cycle_durations_ms: List[float] = []
    if (
        flip_buy_price is not None
        and flip_sell_price is not None
        and np.isfinite(flip_buy_price)
        and np.isfinite(flip_sell_price)
        and flip_buy_price > 0
        and flip_sell_price > 0
    ):
        waiting_for = "buy"
        buy_ts: Optional[float] = None
        for ts, _, _, ah, al, hv, lv in side_pts:
            if waiting_for == "buy":
                if np.isfinite(al) and al > 0 and al <= float(flip_buy_price) and lv > 0:
                    buy_ts = float(ts)
                    waiting_for = "sell"
                continue
            # waiting_for == "sell"
            if np.isfinite(ah) and ah > 0 and ah >= float(flip_sell_price) and hv > 0:
                if buy_ts is not None and np.isfinite(buy_ts) and ts > buy_ts:
                    flip_cycles += 1
                    flip_cycle_durations_ms.append(float(ts - buy_ts))
                buy_ts = None
                waiting_for = "buy"

    flip_cycles_per_day: Optional[float] = None
    if flip_cycles > 0 and denom_days > 0:
        flip_cycles_per_day = float(flip_cycles / denom_days)

    flip_cycle_median_hours: Optional[float] = None
    if flip_cycle_durations_ms:
        med_ms = float(np.median(np.array(flip_cycle_durations_ms, dtype="float64")))
        if np.isfinite(med_ms) and med_ms > 0:
            flip_cycle_median_hours = float(med_ms / (3600.0 * 1000.0))

    fit = fit_two_gaussian_mixture_log(prices, max_iter=MAX_ITER)
    if fit is None:
        return None

    gamma2 = fit["gamma2"]
    w1 = float(fit["w1"])
    w2 = float(fit["w2"])

    # By default, treat the lower-mean component as baseline and the higher-mean
    # component as spikes. If the higher-mean component dominates the mass,
    # we likely have "reverse peaks" (rare dips) and should treat the dominant
    # cluster as the baseline instead.
    baseline_is_comp2 = w2 > w1

    def robust_trimmed_mean(x: np.ndarray) -> Optional[float]:
        if x.size == 0:
            return None
        if x.size < 10:
            return float(np.median(x))
        lo = float(np.quantile(x, 0.05))
        hi = float(np.quantile(x, 0.95))
        trimmed = x[(x >= lo) & (x <= hi)]
        if trimmed.size == 0:
            trimmed = x
        return float(trimmed.mean())

    if baseline_is_comp2:
        baseline_mask = gamma2 > 0.5
        mu_baseline = float(fit["mu2"])
        mu_spike = float(fit["mu1"])
    else:
        baseline_mask = gamma2 <= 0.5
        mu_baseline = float(fit["mu1"])
        mu_spike = float(fit["mu2"])

    baseline_prices = prices[baseline_mask]
    spike_prices = prices[~baseline_mask]

    low_avg = robust_trimmed_mean(baseline_prices)
    if low_avg is None or not np.isfinite(low_avg) or low_avg <= 0:
        low_avg = exp(mu_baseline)

    peak_avg = robust_trimmed_mean(spike_prices)
    if peak_avg is None or not np.isfinite(peak_avg) or peak_avg <= 0:
        peak_avg = exp(mu_spike)

    if not np.isfinite(low_avg) or low_avg <= 0 or not np.isfinite(peak_avg):
        return None

    separation = peak_avg / low_avg
    pct_diff = (separation - 1.0) * 100.0

    def compute_local_mean(ts: np.ndarray, ys: np.ndarray, half_window_ms: float) -> np.ndarray:
        out = np.full(ys.shape, np.nan, dtype="float64")
        if ys.size == 0:
            return out

        left = 0
        right = 0
        window_sum = 0.0

        for i in range(ys.size):
            center = float(ts[i])
            right_bound = center + half_window_ms
            left_bound = center - half_window_ms

            while right < ys.size and float(ts[right]) <= right_bound:
                window_sum += float(ys[right])
                right += 1

            while left < ys.size and float(ts[left]) < left_bound:
                window_sum -= float(ys[left])
                left += 1

            count = right - left
            if count > 0:
                out[i] = window_sum / count
        return out

    # Count peaks using a hysteresis band around a *local* baseline:
    # - Baseline(t) = mean price within ±BASELINE_HALF_WINDOW_DAYS days of t
    # - A peak starts when price >= Baseline(t) * 3.0 (+200%).
    # - Once "in a peak", it doesn't end until price <= Baseline(t) * 1.1 (+10%).
    half_window_ms = float(BASELINE_HALF_WINDOW_DAYS) * 86400.0 * 1000.0
    local_mean = compute_local_mean(ts_ms, prices, half_window_ms=half_window_ms)
    ratios = np.where(np.isfinite(local_mean) & (local_mean > 0), prices / local_mean, np.nan)
    min_peak_ratio = 3.0
    peak_end_ratio = 1.1

    peaks: List[Dict[str, float]] = []
    in_peak = False
    current_peak_max_ratio: Optional[float] = None
    current_peak_max_price: Optional[float] = None
    current_peak_max_ts: Optional[float] = None
    current_peak_max_baseline: Optional[float] = None
    current_peak_start_idx: Optional[int] = None
    current_peak_end_idx: Optional[int] = None

    def peak_avg_exceeds_surrounding(
        start_idx: Optional[int],
        end_idx: Optional[int],
        factor: float = 2.0,
    ) -> bool:
        if start_idx is None or end_idx is None:
            return False
        if start_idx < 0 or end_idx < start_idx or end_idx >= int(prices.size):
            return False

        width = end_idx - start_idx + 1
        if width <= 0:
            return False

        peak_slice = prices[start_idx : end_idx + 1]
        if peak_slice.size == 0:
            return False
        peak_sum = float(np.sum(peak_slice))
        peak_avg = peak_sum / float(peak_slice.size)

        left_start = max(0, start_idx - width)
        left_slice = prices[left_start:start_idx]
        right_end = min(int(prices.size), end_idx + 1 + width)
        right_slice = prices[end_idx + 1 : right_end]
        surround_count = int(left_slice.size) + int(right_slice.size)
        if surround_count <= 0:
            return False
        surround_sum = float(np.sum(left_slice)) + float(np.sum(right_slice))
        surround_avg = surround_sum / float(surround_count)

        if not np.isfinite(peak_avg) or not np.isfinite(surround_avg) or surround_avg <= 0:
            return False
        if not np.isfinite(factor) or factor <= 0:
            factor = 2.0
        return float(peak_avg) > float(factor) * float(surround_avg)

    def record_peak(start_idx: Optional[int], end_idx: Optional[int]) -> None:
        if not peak_avg_exceeds_surrounding(start_idx, end_idx, factor=2.0):
            return

        if (
            current_peak_max_price is None
            or not np.isfinite(current_peak_max_price)
            or current_peak_max_price <= 0
        ):
            return
        if (
            current_peak_max_baseline is None
            or not np.isfinite(current_peak_max_baseline)
            or current_peak_max_baseline <= 0
        ):
            return

        if current_peak_max_ts is None or not np.isfinite(current_peak_max_ts):
            return

        ratio = float(current_peak_max_price) / float(current_peak_max_baseline)
        if not np.isfinite(ratio) or ratio <= 0:
            return

        pct = (ratio - 1.0) * 100.0
        if not np.isfinite(pct):
            return

        peaks.append(
            {
                "ts": float(current_peak_max_ts),
                "ratio": float(ratio),
                "tip_pct": max(0.0, float(pct)),
            }
        )

    for i, ((ts, price, _), ratio, baseline) in enumerate(zip(pts, ratios, local_mean)):
        if not np.isfinite(ratio) or not np.isfinite(baseline) or baseline <= 0:
            continue
        if not in_peak:
            if ratio >= min_peak_ratio:
                in_peak = True
                current_peak_max_ratio = float(ratio)
                current_peak_max_price = price
                current_peak_max_ts = ts
                current_peak_max_baseline = float(baseline)
                current_peak_start_idx = i
                current_peak_end_idx = i
            continue

        if ratio > peak_end_ratio:
            current_peak_end_idx = i
        if current_peak_max_ratio is None or ratio > current_peak_max_ratio:
            current_peak_max_ratio = float(ratio)
            current_peak_max_price = price
            current_peak_max_ts = ts
            current_peak_max_baseline = float(baseline)

        if ratio <= peak_end_ratio:
            record_peak(current_peak_start_idx, current_peak_end_idx)
            in_peak = False
            current_peak_max_ratio = None
            current_peak_max_price = None
            current_peak_max_ts = None
            current_peak_max_baseline = None
            current_peak_start_idx = None
            current_peak_end_idx = None

    # If we end the window while still in a peak, we intentionally do NOT count it.
    # (We haven't observed the peak returning back below the end threshold yet.)

    # If multiple peaks occur within the same 3-day period, keep only the largest
    # (highest peak ratio) and discard the rest.
    peaks.sort(key=lambda p: p.get("ts", 0.0))
    keep_window_ms = float(BASELINE_HALF_WINDOW_DAYS) * 86400.0 * 1000.0
    if keep_window_ms > 0 and len(peaks) > 1:
        deduped: List[Dict[str, float]] = []
        for p in peaks:
            if not deduped:
                deduped.append(p)
                continue
            last = deduped[-1]
            if float(p["ts"]) - float(last["ts"]) <= keep_window_ms:
                if float(p["ratio"]) > float(last["ratio"]):
                    deduped[-1] = p
                continue
            deduped.append(p)
        peaks = deduped

    peaks_count = len(peaks)

    ms_per_day = 86400 * 1000.0
    time_since_last_peak_days: Optional[float] = None
    avg_time_between_peaks_days: Optional[float] = None
    if peaks:
        peak_ts_list = [float(p["ts"]) for p in peaks]
        last_peak_ts = peak_ts_list[-1]
        time_since_last_peak_days = (now_ms - last_peak_ts) / ms_per_day
        if peaks_count >= 2:
            gaps = (
                np.diff(np.array(peak_ts_list, dtype="float64")) / ms_per_day
            )
            if gaps.size:
                avg_time_between_peaks_days = float(np.mean(gaps))
        elif peaks_count == 1:
            avg_time_between_peaks_days = 1000.0

    tip_pct_list = [float(p["tip_pct"]) for p in peaks if np.isfinite(p.get("tip_pct", np.nan))]
    score = float(np.mean(tip_pct_list)) if tip_pct_list else 0.0

    volume24h = 0.0
    for ts, _, vol in pts:
        if ts >= cutoff24h_ms:
            volume24h += vol

    return {
        "low_avg_price": low_avg,
        "peak_avg_price": peak_avg,
        "pct_difference": pct_diff,
        "variance": float(variance) if np.isfinite(variance) else None,
        "variance_pct": float(variance_pct) if variance_pct is not None and np.isfinite(variance_pct) else None,
        "volume_24h": volume24h,
        "peaks_count": peaks_count,
        "time_since_last_peak_days": time_since_last_peak_days,
        "avg_time_between_peaks_days": avg_time_between_peaks_days,
        "score": float(score),
        "flip_buy_price": float(flip_buy_price) if flip_buy_price is not None and np.isfinite(flip_buy_price) else None,
        "flip_sell_price": float(flip_sell_price) if flip_sell_price is not None and np.isfinite(flip_sell_price) else None,
        "flip_edge_gp": float(flip_edge_gp) if flip_edge_gp is not None and np.isfinite(flip_edge_gp) else None,
        "flip_edge_pct": float(flip_edge_pct) if flip_edge_pct is not None and np.isfinite(flip_edge_pct) else None,
        "flip_tail_buy_units_per_day": float(flip_tail_buy_units_per_day) if np.isfinite(flip_tail_buy_units_per_day) else None,
        "flip_tail_sell_units_per_day": float(flip_tail_sell_units_per_day) if np.isfinite(flip_tail_sell_units_per_day) else None,
        "flip_flow_balance_pct": float(flip_flow_balance_pct) if flip_flow_balance_pct is not None and np.isfinite(flip_flow_balance_pct) else None,
        "flip_units_per_day": float(flip_units_per_day) if flip_units_per_day is not None and np.isfinite(flip_units_per_day) else None,
        "flip_cap_units_per_day": float(flip_cap_units_per_day) if flip_cap_units_per_day is not None and np.isfinite(flip_cap_units_per_day) else None,
        "flip_expected_gp_per_day": float(flip_expected_gp_per_day) if flip_expected_gp_per_day is not None and np.isfinite(flip_expected_gp_per_day) else None,
        "spread_pct_mean": float(spread_pct_mean) if spread_pct_mean is not None and np.isfinite(spread_pct_mean) else None,
        "flip_cycles_per_day": float(flip_cycles_per_day) if flip_cycles_per_day is not None and np.isfinite(flip_cycles_per_day) else None,
        "flip_cycle_median_hours": float(flip_cycle_median_hours) if flip_cycle_median_hours is not None and np.isfinite(flip_cycle_median_hours) else None,
    }


def load_latest_mapping(s3, bucket) -> Dict[int, Dict[str, Any]]:
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
            out: Dict[int, Dict[str, Any]] = {}
            for m in mapping:
                try:
                    item_id = int(m.get("id"))
                    name = m.get("name")
                except Exception:
                    continue
                limit_val = None
                try:
                    limit_val = m.get("limit")
                except Exception:
                    limit_val = None
                limit_int = None
                if limit_val is not None:
                    try:
                        limit_int = int(limit_val)
                    except Exception:
                        limit_int = None
                if item_id and isinstance(name, str) and name:
                    out[item_id] = {"name": name, "limit": limit_int}
            if out:
                print("Loaded mapping from", latest_key, "(", len(out), "items )")
            return out
    return {}


def load_latest_daily_volumes(s3, bucket) -> Dict[int, float]:
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

        volumes = data.get("volumes_24h")
        if isinstance(volumes, dict):
            volumes_data = volumes.get("data")
        else:
            volumes_data = None

        if isinstance(volumes_data, dict):
            out: Dict[int, float] = {}
            for item_id_raw, v in volumes_data.items():
                try:
                    item_id = int(item_id_raw)
                except Exception:
                    continue
                if not item_id:
                    continue
                try:
                    vol_f = float(v)
                except Exception:
                    continue
                if np.isfinite(vol_f):
                    out[item_id] = vol_f
            if out:
                print("Loaded volumes from", latest_key, "(", len(out), "items )")
            return out
    return {}


def main():
    s3 = get_r2_client()
    bucket = os.environ["R2_BUCKET"]

    mapping = load_latest_mapping(s3, bucket)
    volumes24h_by_id = load_latest_daily_volumes(s3, bucket)

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
            base = key.rsplit("/", 1)[-1].split(".", 1)[0]
            item_id = int(base)
            map_entry = mapping.get(item_id) if isinstance(mapping, dict) else None
            name = (
                (map_entry.get("name") if isinstance(map_entry, dict) else None)
                or data.get("name")
                or f"Item {item_id}"
            )
            limit_4h = None
            if isinstance(map_entry, dict):
                try:
                    limit_raw = map_entry.get("limit")
                except Exception:
                    limit_raw = None
                try:
                    limit_4h = float(limit_raw) if limit_raw is not None else None
                except Exception:
                    limit_4h = None

            metric = compute_catching_peaks_metric(
                history,
                window_days=WINDOW_DAYS,
                trade_limit_4h=limit_4h,
                tax_rate=TAX_RATE,
            )
            if metric is None:
                return None

            daily_vol = volumes24h_by_id.get(item_id)
            if daily_vol is not None and np.isfinite(daily_vol):
                metric["volume_24h"] = float(daily_vol)

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
        "baseline_half_window_days": BASELINE_HALF_WINDOW_DAYS,
        "tax_rate": TAX_RATE,
        "flip_buy_quantile": FLIP_BUY_Q,
        "flip_sell_quantile": FLIP_SELL_Q,
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
