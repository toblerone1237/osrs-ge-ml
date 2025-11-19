// OSRS GE UI worker: top trades, search, and price history + forecast (using precomputed histories)

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>OSRS GE – ML Trades & Price Forecast</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { color-scheme: dark; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0; padding: 0;
      background: #020617; color: #e5e7eb;
    }
    header { background: #020617; border-bottom: 1px solid #1f2937; padding: 1.1rem 1.4rem; }
    header h1 { margin: 0; font-size: 1.4rem; }
    header p { margin: 0.25rem 0 0; font-size: 0.85rem; color: #94a3b8; }
    main { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1rem; padding: 1rem; }
    section { background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; padding: 1rem; }
    h2 { margin: 0 0 0.75rem 0; font-size: 1.05rem; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 0.5rem; text-align: left; border-bottom: 1px solid #1f2937; }
    th { font-weight: 600; color: #cbd5e1; }
    tr.clickable:hover { background: #0f172a; cursor: pointer; }
    .pin-btn { background: transparent; border: 1px solid #334155; border-radius: 4px; color: #e2e8f0; padding: 2px 6px; }
    .pin-btn.pinned { border-color: #facc15; color: #facc15; }
    .meta { font-size: 0.85rem; color: #94a3b8; }
    .status { font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }
    .warning { color: #fbbf24; }
    .danger { color: #f87171; }
    #chart-container { height: 360px; }
    .small { font-size: 0.85rem; color: #9ca3af; }
  </style>
</head>
<body>
  <header>
    <h1>OSRS GE – ML Trades & Price Forecast</h1>
    <p class="meta" id="meta"></p>
  </header>

  <main>
    <section>
      <h2>Top trades (clean slice)</h2>
      <div id="status" class="status"></div>
      <div id="table-container"></div>
      <p class="small">Items flagged as <span class="warning">noisy</span> (e.g. very cheap with huge predicted returns or extreme Win %) are excluded here to avoid regime-shift artifacts.</p>
    </section>

    <section>
      <h2 id="price-title">Price panel</h2>
      <div id="price-status" class="status"></div>
      <div id="chart-container"><canvas id="chart"></canvas></div>
      <div id="pinned-list"></div>
    </section>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.umd.min.js"></script>

  <script>
  (function () {
    const tableContainer = document.getElementById("table-container");
    const statusEl = document.getElementById("status");
    const metaEl = document.getElementById("meta");
    const priceTitleEl = document.getElementById("price-title");
    const priceStatusEl = document.getElementById("price-status");
    const chartCanvas = document.getElementById("chart");
    const PIN_KEY = "osrs-ge-pinned-v1";

    let overviewSignals = [];
    let dailySnapshot = null;
    let mappingList = [];
    let MODEL_HORIZON = 60;
    let MODEL_TAX = 0.02;
    let priceChart = null;

    function toPct(x) {
      const v = Number(x);
      return Number.isFinite(v) ? (v * 100).toFixed(0) + "%" : "-";
    }

    // --- pin state helpers ---
    function loadPinnedState() {
      try {
        const raw = window.localStorage.getItem(PIN_KEY);
        if (!raw) return {};
        const obj = JSON.parse(raw);
        if (obj && typeof obj === "object") return obj;
      } catch {}
      return {};
    }
    function savePinnedState(state) {
      try { window.localStorage.setItem(PIN_KEY, JSON.stringify(state)); } catch {}
    }
    function getPinnedSet() {
      const state = loadPinnedState();
      return new Set(Object.keys(state).filter((k) => state[k] && state[k].pinned));
    }
    function togglePin(itemId, name, forecastPath) {
      const state = loadPinnedState();
      const key = String(itemId);
      const nowIso = new Date().toISOString();
      const already = state[key] && state[key].pinned;
      state[key] = already
        ? { pinned: false }
        : {
            pinned: true,
            pinnedAtIso: nowIso,
            starredAtIso: nowIso,
            forecastAtStar: Array.isArray(forecastPath) ? forecastPath : []
          };
      savePinnedState(state);
      renderPinnedList();
      renderTopTable();
    }

    // --- mapping from /daily (names) ---
    function buildMappingFromDaily(daily) {
      try {
        const items = daily.items || [];
        mappingList = items.map((it) => ({ id: String(it.id), name: it.name }));
      } catch {
        mappingList = [];
      }
    }
    function nameForId(id) {
      const s = overviewSignals.find((r) => String(r.item_id) === String(id));
      if (s && s.name) return s.name;
      const ent = mappingList.find((m) => String(m.id) === String(id));
      return ent ? ent.name : "Item " + id;
    }

    // --- Top table ---
    function renderTopTable() {
      if (!overviewSignals.length) {
        tableContainer.textContent = "No signals available.";
        return;
      }

      const pins = getPinnedSet();

      const rows = overviewSignals
        // Exclude noisy from the main list
        .filter((s) => !(s && s.noisy === true))
        .map((s) => {
          const id = s.item_id;
          const name = s.name || ("Item " + id);

          const prob = typeof s.prob_profit === "number" ? Math.min(Math.max(s.prob_profit, 0), 1) : 0;
          const profit = typeof s.expected_profit === "number" ? Math.max(0, s.expected_profit) : 0;

          const liqRaw = typeof s.volume_window === "number" ? s.volume_window : 0;
          const liqScore = Math.log1p(Math.max(0, liqRaw));
          const liq = Math.max(1e-3, liqScore || 1);

          const evProfit = prob * profit;
          const probBoost = 0.5 + 0.5 * prob; // mild shaping
          const score = evProfit * liq * probBoost;

          const gpPerHour = (typeof s.expected_profit_per_second === "number" && Number.isFinite(s.expected_profit_per_second))
            ? s.expected_profit_per_second * 3600
            : null;

          return {
            raw: s,
            id,
            name,
            winProb: prob,
            profit,
            gpPerHour,
            volWindow: liqRaw,
            holdMinutes: typeof s.hold_minutes === "number" ? s.hold_minutes : null,
            combinedScore: Number.isFinite(score) ? score : 0,
            pinned: pins.has(String(id))
          };
        })
        .filter((row) => Number.isFinite(row.combinedScore));

      rows.sort((a, b) => b.combinedScore - a.combinedScore);
      const top10 = rows.slice(0, 10);

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const trHead = document.createElement("tr");
      ["★", "Item", "Win %", "Exp. profit (gp)", "GP/hr @limit", "Window volume", "Hold (m)"].forEach((h) => {
        const th = document.createElement("th"); th.textContent = h; trHead.appendChild(th);
      });
      thead.appendChild(trHead);

      top10.forEach((row) => {
        const tr = document.createElement("tr");
        tr.className = "clickable";

        const tdStar = document.createElement("td");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pin-btn " + (row.pinned ? "pinned" : "unpinned");
        btn.textContent = row.pinned ? "★" : "☆";
        btn.addEventListener("click", (ev) => { ev.stopPropagation(); togglePin(row.id, row.name, row.raw.path || []); });
        tdStar.appendChild(btn);
        tr.appendChild(tdStar);

        const tdName = document.createElement("td"); tdName.textContent = row.name; tr.appendChild(tdName);
        const tdWin = document.createElement("td"); tdWin.textContent = toPct(row.winProb); tr.appendChild(tdWin);
        const tdProfit = document.createElement("td"); tdProfit.textContent = Math.max(0, row.profit).toFixed(0); tr.appendChild(tdProfit);
        const tdGph = document.createElement("td"); tdGph.textContent = row.gpPerHour != null ? row.gpPerHour.toFixed(0) : "-"; tr.appendChild(tdGph);
        const tdVol = document.createElement("td"); tdVol.textContent = row.volWindow; tr.appendChild(tdVol);
        const tdHold = document.createElement("td"); tdHold.textContent = row.holdMinutes != null ? row.holdMinutes : "-"; tr.appendChild(tdHold);

        tr.addEventListener("click", () => { loadPriceSeries(row.id, row.name); });
        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);
      tableContainer.innerHTML = "";
      tableContainer.appendChild(table);
    }

    function renderPinnedList() {
      const pinnedState = loadPinnedState();
      const keys = Object.keys(pinnedState).filter((k) => pinnedState[k] && pinnedState[k].pinned);
      const pinnedListEl = document.getElementById("pinned-list");

      if (!keys.length) {
        pinnedListEl.innerHTML = "<p class='small'>No pinned items yet.</p>";
        return;
      }

      const rows = keys.map((k) => {
        const id = Number(k);
        const s = overviewSignals.find((r) => String(r.item_id) === String(id));
        const name = s && s.name ? s.name : nameForId(id);
        const forecastLen = s && Array.isArray(s.path) ? s.path.length : 0;
        const pinnedAtIso = pinnedState[k].pinnedAtIso || pinnedState[k].starredAtIso || null;
        return { item_id: id, name, forecastLen, pinnedAtIso };
      }).sort((a, b) => (b.pinnedAtIso || "").localeCompare(a.pinnedAtIso || ""));

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");
      const trHead = document.createElement("tr");
      ["Item", "Pinned at", "Forecast pts"].forEach((h) => { const th = document.createElement("th"); th.textContent = h; trHead.appendChild(th); });
      thead.appendChild(trHead);

      rows.forEach((row) => {
        const tr = document.createElement("tr"); tr.className = "clickable";
        tr.addEventListener("click", () => loadPriceSeries(row.item_id, row.name));
        const tdName = document.createElement("td"); tdName.textContent = row.name; tr.appendChild(tdName);
        const tdPinned = document.createElement("td"); tdPinned.textContent = row.pinnedAtIso || "unknown"; tr.appendChild(tdPinned);
        const tdN = document.createElement("td"); tdN.textContent = row.forecastLen; tr.appendChild(tdN);
        tbody.appendChild(tr);
      });

      table.appendChild(thead); table.appendChild(tbody);
      pinnedListEl.innerHTML = ""; pinnedListEl.appendChild(table);
    }

    function buildTimeline(history, forecast, starInfo) {
      const parseIso = (iso) => { const t = Date.parse(iso); return Number.isFinite(t) ? t : null; };
      const histMap = new Map(), fcMap = new Map(), oldFcMap = new Map();
      let nowTs = null, starTs = null;

      (history || []).forEach((pt) => {
        const ts = parseIso(pt.timestamp_iso || pt.timestamp); if (ts != null && Number.isFinite(pt.price) && pt.price > 0) histMap.set(ts, pt.price);
      });
      (forecast || []).forEach((pt) => {
        const ts = parseIso(pt.timestamp_iso || pt.timestamp); if (ts != null && Number.isFinite(pt.price) && pt.price > 0) fcMap.set(ts, pt.price);
      });

      if (forecast && forecast.length) {
        const last = forecast[0]; const ts = parseIso(last.timestamp_iso || last.timestamp);
        if (ts != null) nowTs = ts;
      }
      if (starInfo && starInfo.starredAtIso) {
        const ts = parseIso(starInfo.starredAtIso); if (ts != null) starTs = ts;
        (starInfo.forecastAtStar || []).forEach((pt) => {
          const t = parseIso(pt.timestamp_iso || pt.timestamp); if (t != null && Number.isFinite(pt.price) && pt.price > 0) oldFcMap.set(t, pt.price);
        });
      }

      const allTs = new Set([...histMap.keys(), ...fcMap.keys(), ...oldFcMap.keys(), ...(starTs ? [starTs] : []), ...(nowTs ? [nowTs] : [])]);
      const tsList = Array.from(allTs).sort((a, b) => a - b);
      return {
        labels: tsList.map((ts) => new Date(ts)),
        histData: tsList.map((ts) => (histMap.has(ts) ? histMap.get(ts) : null)),
        fcData: tsList.map((ts) => ((nowTs == null || ts >= nowTs) && fcMap.has(ts) ? fcMap.get(ts) : null)),
        oldFcData: tsList.map((ts) => ((starTs == null || ts >= starTs) && oldFcMap.has(ts) ? oldFcMap.get(ts) : null)),
        starMarkerData: tsList.map((ts) => (starTs != null && ts === starTs ? (histMap.get(ts) || oldFcMap.get(ts) || null) : null)),
        nowMarkerData: tsList.map((ts) => (nowTs != null && ts === nowTs ? (histMap.get(ts) || null) : null)),
      };
    }

    async function loadPriceSeries(itemId, name) {
      priceTitleEl.textContent = "Price for " + name + " (id " + itemId + ")";
      priceStatusEl.textContent = "Loading price series...";

      try {
        const res = await fetch("/price-series?item_id=" + encodeURIComponent(itemId));
        if (!res.ok) {
          priceStatusEl.textContent = "No price data available (HTTP " + res.status + ").";
          if (priceChart) { priceChart.destroy(); priceChart = null; }
          return;
        }

        const data = await res.json();
        const history = data.history || [];
        const forecast = data.forecast || [];
        const noisy = !!data.noisy;
        const noiseReason = data.noise_reason || "";

        if (!history.length && !forecast.length) {
          priceStatusEl.textContent = "No price data yet for this item.";
          if (priceChart) { priceChart.destroy(); priceChart = null; }
          return;
        }

        const pinnedState = loadPinnedState();
        const pinEntry = pinnedState[String(itemId)];
        const starInfo = pinEntry && pinEntry.pinned
          ? { starredAtIso: pinEntry.starredAtIso || pinEntry.pinnedAtIso, forecastAtStar: pinEntry.forecastAtStar || [] }
          : null;

        const tl = buildTimeline(history, forecast, starInfo);
        const labels = tl.labels, histData = tl.histData, fcData = tl.fcData, oldFcData = tl.oldFcData, starMarkerData = tl.starMarkerData, nowMarkerData = tl.nowMarkerData;

        const allPrices = [].concat(histData, fcData).filter((v) => v != null && Number.isFinite(v));
        let yMin = 0, yMax = 1;
        if (allPrices.length) {
          let rawMin = Math.min.apply(null, allPrices), rawMax = Math.max.apply(null, allPrices);
          const mid = (rawMin + rawMax) / 2, maxSpreadFactor = 50;
          if (mid > 0 && rawMax / mid > maxSpreadFactor) rawMax = mid * maxSpreadFactor;
          if (mid > 0 && mid / rawMin > maxSpreadFactor) rawMin = mid / maxSpreadFactor;
          const pad = (rawMax - rawMin) * 0.1 || mid * 0.1 || 1; yMin = Math.max(0, rawMin - pad); yMax = rawMax + pad;
        }

        const ctx = chartCanvas.getContext("2d");
        if (priceChart) priceChart.destroy();

        const datasets = [
          { label: "Historical mid price (5m last 24h, 30m older)", data: histData, borderColor: "rgba(59,130,246,1)", backgroundColor: "rgba(59,130,246,0.2)", pointRadius: 0, borderWidth: 2, tension: 0.15, spanGaps: true },
          { label: "Forecast price (next 2h, 5m steps)", data: fcData, borderColor: "rgba(16,185,129,1)", backgroundColor: "rgba(16,185,129,0.15)", pointRadius: 0, borderWidth: 2, borderDash: [6, 3], tension: 0.15, spanGaps: true },
        ];
        if (starInfo && oldFcData.some((v) => v != null)) {
          datasets.splice(1, 0, { label: "Forecast at pin time", data: oldFcData, borderColor: "rgba(234,179,8,1)", backgroundColor: "rgba(234,179,8,0.15)", pointRadius: 0, borderWidth: 1.5, borderDash: [6, 4], tension: 0.15, spanGaps: true });
        }
        if (starInfo && starMarkerData.some((v) => v != null)) {
          datasets.push({ label: "Pin time", data: starMarkerData, borderColor: "rgba(250,204,21,1)", backgroundColor: "rgba(250,204,21,1)", pointRadius: 4, borderWidth: 0, showLine: false });
        }
        if (nowMarkerData.some((v) => v != null)) {
          datasets.push({ label: "Now", data: nowMarkerData, borderColor: "rgba(248,250,252,1)", backgroundColor: "rgba(248,250,252,1)", pointRadius: 4, borderWidth: 0, showLine: false });
        }

        priceChart = new Chart(ctx, {
          type: "line",
          data: { labels, datasets },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
              x: { type: "time", time: { unit: "hour", stepSize: 1, displayFormats: { hour: "MM-dd HH:mm" } }, ticks: { maxRotation: 0, autoSkip: true } },
              y: { beginAtZero: false, suggestedMin: yMin, suggestedMax: yMax,
                   ticks: { callback: (value) => { const v = Number(value)||0;
                     if (v >= 1_000_000_000) return (v/1_000_000_000).toFixed(1)+"b";
                     if (v >= 1_000_000) return (v/1_000_000).toFixed(1)+"m";
                     if (v >= 1_000) return (v/1_000).toFixed(1)+"k"; return v.toFixed(0); } } }
            },
            plugins: {
              legend: { position: "bottom" },
              zoom: {
                zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "x" },
                pan: { enabled: true, mode: "x", modifierKey: "shift" },
                limits: { x: { min: "original", max: "original" } }
              }
            }
          }
        });

        const src = data.meta && data.meta.source ? data.meta.source : "precomputed";
        const truncated = data.meta && typeof data.meta.truncated === "boolean" ? data.meta.truncated : false;
        const hasForecast = forecast && forecast.length > 1;

        if (noisy) {
          const msg = "This instrument is flagged as high-volatility / experimental: " + (noiseReason || "noisy regime");
          priceStatusEl.innerHTML = "<span class='warning'>" + msg + "</span>. History source: " + src + (truncated ? " (truncated)" : "") + ".";
        } else if (!hasForecast) {
          priceStatusEl.textContent = "No ML forecast for this item (no entry in the latest /signals snapshot). Showing history only. History source: " + src + (truncated ? " (truncated)" : "") + ".";
        } else {
          priceStatusEl.textContent = "History source: " + src + (truncated ? " (truncated)" : "") + ". Blue = history; green = current forecast (5–120 minute horizons).";
        }
      } catch (err) {
        console.error("Error loading price series:", err);
        priceStatusEl.textContent = "Error loading price series.";
        if (priceChart) { priceChart.destroy(); priceChart = null; }
      }
    }

    async function loadOverview() {
      try {
        statusEl.textContent = "Fetching /signals and /daily...";
        const [sigRes, dailyRes] = await Promise.all([fetch("/signals"), fetch("/daily")]);

        if (!sigRes.ok) { statusEl.textContent = "Failed to load signals (HTTP " + sigRes.status + ")."; return; }
        const sigJson = await sigRes.json();
        overviewSignals = Array.isArray(sigJson.signals) ? sigJson.signals : [];
        MODEL_HORIZON = typeof sigJson.horizon_minutes === "number" ? sigJson.horizon_minutes : 60;
        MODEL_TAX = typeof sigJson.tax_rate === "number" ? sigJson.tax_rate : 0.02;

        if (dailyRes.ok) { dailySnapshot = await dailyRes.json(); buildMappingFromDaily(dailySnapshot); }
        else { dailySnapshot = null; mappingList = []; }

        statusEl.textContent = "";
        metaEl.textContent = "Signals computed at " + (sigJson.generated_at_iso || "unknown time") + " – horizon " + MODEL_HORIZON + " minutes, tax " + (MODEL_TAX * 100).toFixed(1) + "%.";
        renderTopTable();
        renderPinnedList();
      } catch (err) {
        console.error("Error loading overview:", err);
        statusEl.textContent = "Error loading overview.";
      }
    }

    loadOverview();
  })();
  </script>
</body>
</html>`;

// ------------------------
// Worker runtime (backend)
// ------------------------

const SIGNALS_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
let LAST_SIGNALS_JSON = null;
let LAST_SIGNALS_FETCHED_AT = 0;

const PRICE_CACHE = new Map();
const PRICE_CACHE_TTL_MS = 5 * 60 * 1000;
const PRICE_CACHE_MAX_ITEMS = 64;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function bucketGetWithRetry(env, key, { attempts = 3, baseDelayMs = 200 } = {}) {
  let lastError = null;
  for (let i = 0; i < attempts; i++) {
    try { return await env.OSRS_BUCKET.get(key); }
    catch (err) {
      lastError = err;
      const isLast = i === attempts - 1;
      const delay = baseDelayMs * Math.pow(2, i) + Math.random() * 100;
      console.warn("Attempt " + (i + 1) + " to fetch " + key + " failed: " + err.message + (isLast ? "" : "; retrying in " + delay + "ms"));
      if (!isLast) await sleep(delay);
    }
  }
  console.error("Failed to fetch " + key + " after " + attempts + " attempts", lastError);
  return null;
}

async function bucketListWithRetry(env, options, { attempts = 3, baseDelayMs = 200 } = {}) {
  let lastError = null;
  for (let i = 0; i < attempts; i++) {
    try { return await env.OSRS_BUCKET.list(options); }
    catch (err) {
      lastError = err;
      const isLast = i === attempts - 1;
      const delay = baseDelayMs * Math.pow(2, i) + Math.random() * 100;
      console.warn("Attempt " + (i + 1) + " to list " + (options && options.prefix ? options.prefix : "") + " failed: " + err.message + (isLast ? "" : "; retrying in " + delay + "ms"));
      if (!isLast) await sleep(delay);
    }
  }
  console.error("Failed to list with options " + JSON.stringify(options) + " after " + attempts + " attempts", lastError);
  return null;
}

async function loadSignalsWithCache(env) {
  const now = Date.now();
  if (LAST_SIGNALS_JSON && now - LAST_SIGNALS_FETCHED_AT < SIGNALS_CACHE_TTL_MS) {
    return { json: LAST_SIGNALS_JSON, source: "cache" };
  }
  const obj = await bucketGetWithRetry(env, "signals/remove-noisy-sections/latest.json");
  if (obj) {
    try {
      const text = await obj.text();
      const parsed = JSON.parse(text);
      if (parsed && Array.isArray(parsed.signals)) {
        LAST_SIGNALS_JSON = parsed;
        LAST_SIGNALS_FETCHED_AT = Date.now();
        return { json: parsed, source: "fresh" };
      }
    } catch (err) { console.error("Error parsing signals/remove-noisy-sections/latest.json:", err); }
  }
  return { json: LAST_SIGNALS_JSON, source: LAST_SIGNALS_JSON ? "stale" : "missing" };
}

function getCachedPriceSeries(cacheKey) {
  const entry = PRICE_CACHE.get(cacheKey);
  if (!entry) return null;
  const age = Date.now() - entry.cachedAt;
  if (age > PRICE_CACHE_TTL_MS) { PRICE_CACHE.delete(cacheKey); return null; }
  return entry.payload;
}
function setCachedPriceSeries(cacheKey, payload) {
  PRICE_CACHE.set(cacheKey, { cachedAt: Date.now(), payload });
  if (PRICE_CACHE.size > PRICE_CACHE_MAX_ITEMS) {
    const oldestKey = PRICE_CACHE.keys().next().value;
    if (oldestKey !== undefined) PRICE_CACHE.delete(oldestKey);
  }
}

async function handleSignals(env) {
  const obj = await bucketGetWithRetry(env, "signals/remove-noisy-sections/latest.json");
  if (!obj) return new Response(JSON.stringify({ error: "No signals found" }), { status: 404, headers: { "Content-Type": "application/json" } });
  const text = await obj.text();
  return new Response(text, { status: 200, headers: { "Content-Type": "application/json" } });
}

// Load precomputed per-item history from R2: history/{itemId}.json
async function loadPrecomputedHistory(env, itemId) {
  const key = "history/" + itemId + ".json";
  const obj = await bucketGetWithRetry(env, key, { attempts: 2, baseDelayMs: 150 });
  if (!obj) { return { history: [], found: false }; }

  try {
    const text = await obj.text();
    const parsed = JSON.parse(text);
    const raw = Array.isArray(parsed.history) ? parsed.history : [];

    const cleaned = raw
      .map((pt) => {
        const iso = pt.timestamp_iso || pt.timestamp || pt.time_iso || pt.time || null;
        const ts = iso ? Date.parse(iso) : NaN;
        const price = typeof pt.price === "number" && Number.isFinite(pt.price) ? pt.price : NaN;
        return { ts, iso, price };
      })
      .filter((pt) => Number.isFinite(pt.ts) && pt.iso && typeof pt.iso === "string" && Number.isFinite(pt.price) && pt.price > 0)
      .sort((a, b) => a.ts - b.ts)
      .map((pt) => ({ timestamp_iso: new Date(pt.ts).toISOString(), price: pt.price }));

    return { history: cleaned, found: cleaned.length > 0 };
  } catch (err) {
    console.error("Failed to parse precomputed history", key, err);
    return { history: [], found: false };
  }
}

// Build forecast path from signals, anchored to last history price (or mid_now)
async function buildForecast(env, itemId, history) {
  const forecast = [];
  const { json: sigJson } = await loadSignalsWithCache(env);
  if (!sigJson) return forecast;

  const signals = sigJson.signals || [];
  const s = signals.find((row) => row.item_id === itemId || String(row.item_id) === String(itemId));
  if (!s || !Array.isArray(s.path) || typeof s.mid_now !== "number") return forecast;

  const anchorMid = history.length ? history[history.length - 1].price : s.mid_now;
  if (!Number.isFinite(anchorMid) || anchorMid <= 0) return forecast;

  let baseTimeMs;
  if (history.length) {
    const last = history[history.length - 1];
    const iso = last.timestamp_iso || last.timestamp || null;
    const ts = iso ? Date.parse(iso) : NaN;
    baseTimeMs = Number.isFinite(ts) ? ts : Date.now();
  } else {
    baseTimeMs = Date.now();
  }

  forecast.push({ timestamp_iso: new Date(baseTimeMs).toISOString(), price: anchorMid });

  for (const p of s.path) {
    if (!p || typeof p.minutes !== "number" || typeof p.future_return_hat !== "number") continue;
    let ret = p.future_return_hat;
    ret = Math.max(-0.9, Math.min(3.0, ret));  // guardrails
    const price = anchorMid * (1 + ret);
    const ts = baseTimeMs + p.minutes * 60_000;
    if (Number.isFinite(price) && price > 0 && Number.isFinite(ts)) {
      forecast.push({ timestamp_iso: new Date(ts).toISOString(), price });
    }
  }
  return forecast;
}

async function handlePriceSeries(env, itemId) {
  const cacheKey = String(itemId);
  const cached = getCachedPriceSeries(cacheKey);
  if (cached) return new Response(cached, { status: 200, headers: { "Content-Type": "application/json" } });

  const { history, found } = await loadPrecomputedHistory(env, itemId);
  const forecast = await buildForecast(env, itemId, history);
  const truncated = !found;

  // Augment with noise flag/reason (from the signals payload)
  let noisy = false;
  let noise_reason = "";
  try {
    const { json: sigJson } = await loadSignalsWithCache(env);
    if (sigJson && Array.isArray(sigJson.signals)) {
      const s = sigJson.signals.find((row) => row.item_id === itemId || String(row.item_id) === String(itemId));
      if (s) {
        noisy = !!s.noisy;
        if (typeof s.noise_reason === "string") noise_reason = s.noise_reason;
      }
    }
  } catch {}

  const body = JSON.stringify({
    item_id: itemId,
    history,
    forecast,
    noisy,
    noise_reason,
    meta: {
      truncated: truncated,
      source: found ? "precomputed" : "missing"
    }
  });

  setCachedPriceSeries(cacheKey, body);

  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      ...(truncated ? { "X-Price-Series-Partial": "1" } : {})
    }
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/") {
      return new Response(HTML, { status: 200, headers: { "Content-Type": "text/html; charset=utf-8" } });
    }
    if (url.pathname === "/signals") {
      return handleSignals(env);
    }
    if (url.pathname === "/daily") {
      const now = new Date();
      for (let delta = 0; delta < 7; delta++) {
        const d = new Date(now.getTime() - delta * 86400000);
        const year = d.getUTCFullYear();
        const month = String(d.getUTCMonth() + 1).padStart(2, "0");
        const day = String(d.getUTCDate()).padStart(2, "0");
        const prefix = "daily/" + year + "/" + month + "/" + day;
        const listing = await bucketListWithRetry(env, { prefix, limit: 1000 });
        if (!listing || !listing.objects || !listing.objects.length) continue;
        const objects = listing.objects.slice().sort((a, b) => (a.key < b.key ? -1 : 1));
        const latest = objects[objects.length - 1];
        const obj = await bucketGetWithRetry(env, latest.key);
        if (!obj) continue;
        const text = await obj.text();
        return new Response(text, { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({ error: "No daily snapshot found" }), { status: 404, headers: { "Content-Type": "application/json" } });
    }
    if (url.pathname === "/price-series") {
      const idParam = url.searchParams.get("item_id");
      const itemId = idParam ? Number(idParam) : NaN;
      if (!Number.isFinite(itemId) || itemId <= 0) {
        return new Response(JSON.stringify({ error: "Invalid item_id" }), { status: 400, headers: { "Content-Type": "application/json" } });
      }
      return handlePriceSeries(env, itemId);
    }
    return new Response("Not found", { status: 404 });
  }
};
