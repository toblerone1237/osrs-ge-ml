# Agents Overview
- The system ingests market data from the official OSRS Wiki APIs, stores consolidated snapshots in the Cloudflare R2 bucket `osrs-ge-raw`, trains and scores ML models against that history, and serves forecasts plus visualisations via Cloudflare Workers.
- Each worker or CLI program under `workers/` and `ml/` acts as an agent for a specific autonomous task (data collection, scoring, serving, analytics, or diagnostics). All of them treat the R2 bucket as the source of truth.
- There is no scheduler inside this repo for the Python agents. They must be invoked manually or through external automation even though their code assumes recurring runs.
- Cloudflare Worker cron schedules are commented out in `wrangler.toml` files, so the actual cadence must be configured in the Cloudflare dashboard or via infrastructure tooling.

```
OSRS Wiki API
   ├─> osrs-ge-5m Worker ─┐
   └─> osrs-ge-daily Worker ─┤
                            v
                     R2 bucket (osrs-ge-raw)
                            ├─> ml/train_model.py ─┐
                            │                      ├─> ml/score_latest.py ─> signals/quantile
                            ├─> ml/build_price_history.py ─> history/{item}.json
                            │                      │
                            └─> osrs-ge-ui Worker <┘ (serves /, /signals, /daily, /price-series)
Analytics & QA (ml/analyse_sigma_buckets.py, ml/backtest*.py, ml/check_recent_activity.py) read the same R2 data.
```

## Agent Catalogue

### OSRS GE 5m Snapshot Worker
- **Identifier:** `workers/osrs-ge-5m/src/index.js`, `workers/osrs-ge-5m/wrangler.toml`
- **Purpose:** Fetch the OSRS Wiki `/5m` and `/latest` endpoints every five minutes and persist combined snapshots so downstream jobs have timely trade data.
- **Triggers / Entry Points:** Cloudflare Worker cron (intended `*/5 * * * *`, currently commented) and a manual HTTP GET `/run-once` for ad-hoc runs; default fetch handler responds with diagnostics text.
- **Inputs:** OSRS Wiki REST API, current UTC time.
- **Outputs / Side Effects:** Writes JSON blobs like `5m/YYYY/MM/DD/HH-mm.json` into the `OSRS_BUCKET` binding (Cloudflare R2), embedding both five-minute aggregates and the `latest` highs/lows, plus metadata such as `fetched_at_iso`.
- **External Dependencies:** Cloudflare Workers runtime, Cloudflare R2 bucket `osrs-ge-raw`, OSRS Wiki API, custom `User-Agent`.
- **Configuration:** Hard-coded `USER_AGENT` string should be customised to a real contact; requires the R2 binding named `OSRS_BUCKET`; cron schedule must be set via Cloudflare.
- **Interactions:** Provides the raw data consumed by `ml/train_model.py`, `ml/score_latest.py`, `ml/build_price_history.py`, and the `/price-series` endpoint in `osrs-ge-ui`.
- **Caveats / Notes:** Only exponential-backoff retry logic guards fetches; if OSRS API rejects the `USER_AGENT` the worker will fail until redeployed with a valid value.

### OSRS GE Daily Snapshot Worker
- **Identifier:** `workers/osrs-ge-daily/src/index.js`, `workers/osrs-ge-daily/wrangler.toml`
- **Purpose:** Capture the daily `/24h`, `/volumes`, and `/mapping` aggregates so item metadata and 24h statistics are available to scorers and the UI.
- **Triggers / Entry Points:** Cloudflare cron (commented `0 0 * * *`) and a basic fetch handler that just returns text for diagnostics.
- **Inputs:** OSRS Wiki REST API endpoints `/24h`, `/volumes`, `/mapping`.
- **Outputs / Side Effects:** Stores snapshots under `daily/YYYY/MM/DD.json` and refreshes `daily/latest.json` in R2 with timing metadata plus the three payloads.
- **External Dependencies:** Same as the 5m worker (Cloudflare Worker runtime, R2 bucket, OSRS API).
- **Configuration:** Uses the same `USER_AGENT` constant and `OSRS_BUCKET` binding; cron must be configured outside the repo.
- **Interactions:** Supplies item mapping (id/name/limit) for `ml/score_latest.py` and `/daily` responses for `osrs-ge-ui`. Without it, names and search functionality degrade.
- **Caveats / Notes:** No retry loop is implemented; a transient API failure would skip that day's file unless the worker is re-run manually.

### OSRS GE UI Worker
- **Identifier:** `workers/osrs-ge-ui/src/index.js`, `workers/osrs-ge-ui/wrangler.toml`
- **Purpose:** Serve the web UI, leaderboard, and JSON APIs that expose signals, daily data, and per-item price histories plus forecasts.
- **Triggers / Entry Points:** Cloudflare Worker HTTP fetches at `/`, `/signals`, `/daily`, and `/price-series?item_id=`.
- **Inputs:** Reads `signals/quantile/latest.json`, `daily/YYYY/MM/DD*.json`, `history/{item_id}.json`, and the most recent `5m` snapshot from R2.
- **Outputs / Side Effects:** Returns HTML (with Chart.js visualisation, watchlists, etc.) and JSON for `/signals`, `/daily`, and `/price-series`. Maintains in-memory caches for signals (10 min), per-item price-series results (5 min, capped at 64 items), and the last `5m` snapshot (~2 min).
- **External Dependencies:** Cloudflare Workers runtime, R2 bucket binding `OSRS_BUCKET`, Chart.js + plugins via CDN, browser localStorage for pin state.
- **Configuration:** Cache TTL constants (`SIGNALS_CACHE_TTL_MS`, etc.), `OSRS_BUCKET` binding, Worker name `osrs-ge-ui-quantile`. No feature flags exposed via env.
- **Interactions:** `/price-series` merges precomputed histories from `ml/build_price_history.py` with live 5m data, anchors forecasts built in `ml/score_latest.py`, and exposes search/mapping sourced from `osrs-ge-daily`.
- **Caveats / Notes:** Listing R2 keys for `/price-series` is unbounded (limit 1000) and assumes the latest day has the newest object; large days may require pagination changes.

### Quantile Model Trainer
- **Identifier:** `ml/train_model.py`
- **Purpose:** Train regime-specific multi-horizon XGBoost regressors on the last 30 days of five-minute snapshots, storing both models and metadata back into R2.
- **Triggers / Entry Points:** CLI invocation (`python ml/train_model.py`); no scheduler in repo.
- **Inputs:** Pulls `5m/YYYY/MM/DD/*.json` from R2 via the S3-compatible API (`get_r2_client`); optional env overrides (`TRAIN_END_ISO` or `TRAIN_END_UNIX`) bound the training window.
- **Outputs / Side Effects:** Writes `models/quantile/YYYY/MM/DD/reg_multi.pkl`, updates `models/quantile/latest_reg.pkl`, and emits `meta.json` / `latest_meta.json` with feature lists, regime definitions, sigma estimates, calibration slopes, and return-scaling config.
- **External Dependencies:** Python stack (`boto3`, `pandas`, `numpy`, `xgboost`, `joblib`), Cloudflare R2 credentials exposed via `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_BUCKET`.
- **Configuration:** Constants for horizons, window length, decay, tax/margin assumptions, regime boundaries, return scaling, and minimum sample counts; experiment name fixed to `"quantile"` (affects R2 paths).
- **Interactions:** Produces the artefacts consumed by `ml/score_latest.py`, `ml/analyse_sigma_buckets.py`, and both backtest scripts; metadata also informs regime penalties in the UI.
- **Caveats / Notes:** If a regime lacks at least `MIN_SAMPLES_PER_HORIZON` rows it is silently skipped, meaning downstream scorers need to tolerate missing regimes.

### Latest Snapshot Scorer
- **Identifier:** `ml/score_latest.py`
- **Purpose:** Score the most recent hour of 5m snapshots using the latest models, compute expected profits, win probabilities, and forecast paths, then publish ranked signals.
- **Triggers / Entry Points:** CLI invocation; often run after each model update and whenever fresh signals are needed.
- **Inputs:** Reads `models/quantile/latest_reg.pkl` and `latest_meta.json`, recent `5m` files (selected by filename order), and up to seven days of `daily` snapshots for item metadata. Requires all R2 credentials.
- **Outputs / Side Effects:** Writes `signals/quantile/YYYY/MM/DD.json` plus `signals/quantile/latest.json` containing metadata, per-item metrics (probabilities, horizon, volume window), and per-horizon forecast paths. Logging reports mapping source and counts.
- **External Dependencies:** Same Python + R2 stack as the trainer.
- **Configuration:** Constants such as `HORIZON_MINUTES` (60), `WINDOW_MINUTES` (60), `MIN_VOLUME_WINDOW`, tax/margin assumptions, and Normal-approximation sigma defaults; experiment namespace fixed to `"quantile"`.
- **Interactions:** UI’s `/signals` endpoint serves this file, `/price-series` reuses the per-item `path`, and analysts rely on the metadata for display. `ml/check_recent_activity.py` reuses helper functions from this module.
- **Caveats / Notes:** Requires a recent `daily` snapshot for item names; if none exist, outputs still write but names remain generic. Filtering uses only volume >= 1, so inactive items may appear if they had a single trade.

### Price History Builder
- **Identifier:** `ml/build_price_history.py`
- **Purpose:** Maintain per-item price histories up to 14 days, keeping the last 24 hours at 5-minute resolution and downsampling older data to 30-minute buckets for efficient serving.
- **Triggers / Entry Points:** CLI invocation; meant to be run periodically or after batches of new snapshots arrive.
- **Inputs:** R2 `5m` snapshots (iterating over recent days), existing `history/{item}.json` files, and `history/_meta.json` to resume from the last processed key.
- **Outputs / Side Effects:** Writes `history/{item_id}.json` with ordered `{"timestamp_iso","price"}` arrays and updates `history/_meta.json` (tracks `last_processed_key` and generation time). Updates only files that change.
- **External Dependencies:** Same Python + R2 credential stack; uses `ThreadPoolExecutor` for concurrent downloads/writes.
- **Configuration:** Constants for retention (`MAX_HISTORY_DAYS=14`), resolutions (`RECENT_WINDOW_HOURS=24`, `OLDER_INTERVAL_MIN=30`), worker counts via env (`SNAPSHOT_FETCH_WORKERS`, `HISTORY_FETCH_WORKERS`, `HISTORY_WRITE_WORKERS`).
- **Interactions:** `osrs-ge-ui` relies on these history files to render past prices and only falls back to pure 5m data if missing. The `_meta` checkpoint prevents reprocessing old snapshots.
- **Caveats / Notes:** A missing or stale `_meta` triggers a full rebuild, which can be expensive because it downloads every stored `5m` file. Snapshots are assumed to exist for every 5-minute bucket; gaps produce sparse histories.

### Sigma Bucket Analyzer
- **Identifier:** `ml/analyse_sigma_buckets.py`
- **Purpose:** Evaluate residual error (`sigma`) across regimes, time-of-day, price ranges, and forecast bins to calibrate risk penalties and diagnose hotspots.
- **Triggers / Entry Points:** CLI invocation, usually after new models or significant data collection periods.
- **Inputs:** Pulls the last `EVAL_DAYS` (default 14) of `5m` snapshots, the latest regression artefacts, and metadata from R2.
- **Outputs / Side Effects:** Produces CSV diagnostics locally (`sigma_overall.csv`, `sigma_prob_metrics.csv`, `feature_correlations.csv`, `permutation_importance.csv`, `sigma_buckets_full.csv`, `sigma_regimes.csv`) and uploads them to `analysis/<prefix>/...` in R2. Prefix is determined by `SIGMA_BUCKET_PREFIX`, `SIGMA_PREFIX`, or a branch-based heuristic.
- **External Dependencies:** Same ML stack plus numpy/pandas for metrics, Cloudflare R2 via boto3.
- **Configuration:** Environment vars for permutation sampling (`PERMUTATION_SAMPLE_ROWS`, `PERMUTATION_RANDOM_SEED`), prefix selection, and the inherited tax/margin/regime constants.
- **Interactions:** Results inform `sigma_main_per_regime` tuning in model training and adjustments to `regime_penalty` in `ml/score_latest.py`.
- **Caveats / Notes:** Requires enough rows per bucket (`MIN_ROWS_PER_BUCKET=100`); sparse combinations are dropped, so the output may lack certain time/price bins.

### Backtest Runner (v1)
- **Identifier:** `ml/backtest.py`
- **Purpose:** Replay historical snapshots with the latest model to benchmark fixed policies and slippage assumptions over a user-supplied date range.
- **Triggers / Entry Points:** CLI invocation with `--start-date` and `--end-date` (YYYY-MM-DD); optional `--reg-key` / `--meta-key` select alternate artefacts.
- **Inputs:** 5m snapshots from the requested window, `models/quantile/latest_reg.pkl`, and `latest_meta.json`.
- **Outputs / Side Effects:** Prints tabular stats per policy/slippage combination and writes `backtest_results.csv` summarising trade counts, hit rate, and return percentiles.
- **External Dependencies:** Same Python + R2 stack; requires numpy/pandas/joblib.
- **Configuration:** Built-in policies (baseline, prob thresholds, hotspot blocks) and slippage presets; no env-driven tuning besides R2 credentials.
- **Interactions:** Validates that `ml/score_latest.py` outputs align with realised returns before exposing them via the UI.
- **Caveats / Notes:** Uses a single static model for the whole window, so it does not account for concept drift or training updates during the backtest period.

### Walk-Forward Backtest Runner (v2)
- **Identifier:** `ml/backtest_v2.py`
- **Purpose:** Perform embargoed, walk-forward backtests that retrain per fold, explore additional policy thresholds, and optionally upload CSV outputs to R2 for sharing.
- **Triggers / Entry Points:** CLI invocation with many knobs (`--start-date`, `--end-date`, `--train-window-days`, `--test-window-days`, `--embargo-minutes`, `--anchored`, policy threshold flags, `--slippage-multipliers`, `--upload-prefix`, etc.).
- **Inputs:** 5m snapshots across the requested period, latest metadata for feature lists/return scaling, and (per fold) the training subsets derived from those snapshots.
- **Outputs / Side Effects:** Saves `backtest_v2_trades.csv`, `backtest_v2_folds.csv`, `backtest_v2_summary.csv`, and `backtest_v2_factor_grid.csv` locally; optionally uploads each to `s3://osrs-ge-raw/<upload-prefix>/<name>_<timestamp>.csv`.
- **External Dependencies:** Python ML stack with `xgboost`, pandas, numpy, Cloudflare R2.
- **Configuration:** Command-line flags describe windows, thresholds, slippage multipliers, and upload targets; inherits tax/margin/regime constants from the features module.
- **Interactions:** Offers richer diagnostics for the live scoring policy, including fold-level Sharpe/max drawdown statistics; complements the simpler `ml/backtest.py`.
- **Caveats / Notes:** Computationally heavy—each fold retrains XGBoost models—so it should be run off-cluster or in batch jobs; missing folds (due to sparse data) simply skip output.

### Recent Activity Checker
- **Identifier:** `ml/check_recent_activity.py`
- **Purpose:** Quickly inspect whether specific item IDs have seen trades in the latest hour of snapshots, mainly for debugging dormant items in signals.
- **Triggers / Entry Points:** CLI invocation `python ml/check_recent_activity.py <item_id> [<item_id> ...]`.
- **Inputs:** Uses `score_latest.get_latest_5m_key()` plus `list_keys_with_prefix` to fetch up to the last 12 five-minute files from R2, then flattens them.
- **Outputs / Side Effects:** Prints per-item summaries (bucket counts, total volume, last timestamp, and a short trailing log) to stdout; no files are written.
- **External Dependencies:** Same Python + R2 stack, pandas, helper utilities from `features.py`.
- **Configuration:** Shares the same R2 credential env vars; no additional tuning knobs.
- **Interactions:** Helps operators decide whether missing signals stem from low activity versus scoring issues before rerunning `ml/score_latest.py`.
- **Caveats / Notes:** Only inspects a single day’s prefix; if more than 12 buckets are missing the window may be <60 minutes, so results should be interpreted accordingly.

## Workflows and Interactions
- **Data ingestion:** `osrs-ge-5m` and `osrs-ge-daily` Workers keep R2 populated with fresh high-frequency and daily data. Their cron schedules are defined in comments but must be configured through Cloudflare; otherwise nothing triggers automatically.
- **Historical curation:** `ml/build_price_history.py` turns raw five-minute snapshots into compact `history/{item}.json` files, which the UI uses directly and which serve as the base timeline for per-item forecasts.
- **Model lifecycle:** `ml/train_model.py` periodically refreshes regime-aware regressors and metadata. `ml/analyse_sigma_buckets.py` consumes both historical data and these models to calibrate sigma buckets and feature importance, feeding insights back into the trainer (e.g. adjusting `REGIME_DEFS` or penalties).
- **Forecast publication:** `ml/score_latest.py` combines the latest hour of data with trained models and writes ranked `signals/quantile/*.json`. These signals drive the UI’s leaderboard and supply forecast paths for `/price-series`.
- **Serving layer:** `osrs-ge-ui` reads `signals`, `daily`, `history`, and live 5m snapshots. When `/price-series` is hit, it loads `history/{item}.json`, optionally appends the latest 5m point, and overlays the `path` predicted by `score_latest`. Browser-side JavaScript (Chart.js) renders forecasts and watchlists based on this API.
- **Quality & diagnostics:** `ml/backtest.py`, `ml/backtest_v2.py`, and `ml/analyse_sigma_buckets.py` provide offline validation loops, while `ml/check_recent_activity.py` offers ad-hoc health checks. None of these write data other than CSVs (plus optional R2 uploads), so scheduling and retention must be managed externally.

## Implementation Reference
| Agent | Key files / directories | Notes |
| --- | --- | --- |
| OSRS GE 5m Snapshot Worker | `workers/osrs-ge-5m/src/index.js`, `workers/osrs-ge-5m/wrangler.toml` | Cloudflare Worker writing `5m/` snapshots and exposing `/run-once`. |
| OSRS GE Daily Snapshot Worker | `workers/osrs-ge-daily/src/index.js`, `workers/osrs-ge-daily/wrangler.toml` | Fetches `/24h`, `/volumes`, `/mapping` each day and updates `daily/latest.json`. |
| OSRS GE UI Worker | `workers/osrs-ge-ui/src/index.js`, `workers/osrs-ge-ui/wrangler.toml` | Serves the HTML UI plus `/signals`, `/daily`, `/price-series` APIs backed by R2. |
| Quantile Model Trainer | `ml/train_model.py`, `ml/features.py` | Trains XGBoost models per regime/horizon and stores artefacts in `models/quantile/`. |
| Latest Snapshot Scorer | `ml/score_latest.py` | Scores the recent hour and writes `signals/quantile/*.json` used by the UI and price-series. |
| Price History Builder | `ml/build_price_history.py` | Maintains per-item `history/{id}.json` with 5m + 30m resolution and `_meta` checkpoints. |
| Sigma Bucket Analyzer | `ml/analyse_sigma_buckets.py` | Generates bucketed sigma diagnostics and uploads CSVs to `analysis/<prefix>/`. |
| Backtest Runner (v1) | `ml/backtest.py` | Applies latest model to historical windows for basic policy/slippage evaluation. |
| Walk-Forward Backtest Runner (v2) | `ml/backtest_v2.py` | Walk-forward, embargoed backtests with per-fold retraining and optional R2 uploads. |
| Recent Activity Checker | `ml/check_recent_activity.py` | CLI helper reporting last-60-minute trade activity for specific item IDs. |
