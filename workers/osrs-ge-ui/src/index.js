// OSRS GE UI worker: top trades, search, and price history + forecast (using precomputed histories)

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>OSRS GE – ML Trades & Price Forecast</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      color-scheme: dark;
    }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #020617;
      color: #e5e7eb;
    }
    header {
      background: #020617;
      border-bottom: 1px solid #1f2937;
      padding: 1.1rem 1.4rem;
    }
    header h1 {
      margin: 0;
      font-size: 1.4rem;
    }
    header p {
      margin: 0.25rem 0 0;
      font-size: 0.85rem;
      color: #9ca3af;
    }
    main {
      max-width: 1150px;
      margin: 1.5rem auto 3rem;
      padding: 0 1rem;
    }
    .card {
      background: #020617;
      border-radius: 0.75rem;
      padding: 1rem 1.25rem;
      border: 1px solid #1f2937;
      box-shadow: 0 10px 30px rgba(15,23,42,0.9);
      margin-bottom: 1.25rem;
    }
    h2 {
      margin-top: 0;
      font-size: 1.1rem;
    }
    .small {
      font-size: 0.85rem;
      color: #9ca3af;
    }
    #status, #meta, #priceStatus, #searchStatus {
      font-size: 0.85rem;
      color: #9ca3af;
    }
    #status {
      margin-bottom: 0.4rem;
    }
    #meta {
      margin-top: 0.15rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.35rem 0.45rem;
      border-bottom: 1px solid #1f2937;
      text-align: right;
    }
    th:first-child, td:first-child {
      text-align: left;
    }
    th {
      color: #9ca3af;
      font-weight: 600;
    }
    tr.clickable {
      cursor: pointer;
    }
    tr.clickable:hover {
      background: rgba(55,65,81,0.5);
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        "Liberation Mono", "Courier New", monospace;
    }
    .pill {
      display: inline-block;
      padding: 0.1rem 0.4rem;
      border-radius: 999px;
      font-size: 0.75rem;
      background: #1f2937;
      color: #d1d5db;
    }
    .pill.good {
      background: #064e3b;
      color: #6ee7b7;
    }
    .pill.bad {
      background: #7f1d1d;
      color: #fecaca;
    }
    .pin-btn {
      background: transparent;
      border: none;
      cursor: pointer;
      font-size: 1.1rem;
      line-height: 1;
      padding: 0;
    }
    .pin-btn.pinned {
      color: #facc15;
    }
    .pin-btn.unpinned {
      color: #4b5563;
    }
    .search-row {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    .search-row input {
      flex: 1;
      padding: 0.4rem 0.5rem;
      border-radius: 0.35rem;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
    }
    .search-row button {
      padding: 0.4rem 0.8rem;
      border-radius: 0.35rem;
      border: 1px solid #4b5563;
      background: #111827;
      color: #e5e7eb;
      cursor: pointer;
    }
    .search-row button:hover {
      background: #1f2937;
    }
    .pinned-list table {
      font-size: 0.8rem;
    }
    canvas {
      max-width: 100%;
    }
    @media (max-width: 750px) {
      table { font-size: 0.78rem; }
      header h1 { font-size: 1.2rem; }
      .card { padding: 0.85rem 0.9rem; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <header>
    <h1>OSRS GE – ML Trades &amp; Price Forecast</h1>
    <p>
      Top picks weighted by win probability, expected profit, and 24h liquidity plus recent mid prices and a 5–120 minute price forecast.
      You can pin items to keep a watchlist and use search to jump to any item.
    </p>
  </header>
  <main>
    <div class="card">
      <div id="status">Loading signals &amp; daily snapshot...</div>
      <div id="meta"></div>
    </div>

    <div class="card">
      <h2>Top 10 items by probability-adjusted profit × liquidity</h2>
      <div class="small" style="margin-bottom:0.4rem;">
        Click a row to see up to the last 14 days (or as far back as available) of mid prices:
        5-minute buckets for the last 24 hours, then ~30-minute buckets further back, plus a 5–120 minute forecast.<br/>
        Click the ★ column to pin an item. Pins persist across refreshes.
      </div>
      <div id="tableContainer">Waiting for data...</div>
    </div>

    <div class="card">
      <h2>Search &amp; watchlist</h2>
      <div class="search-row">
        <input id="searchInput" type="text" placeholder="Search by item name..." />
        <button id="searchButton" type="button">Search</button>
      </div>
      <div id="searchStatus" class="small"></div>
      <div id="searchResults" style="margin-top:0.4rem;"></div>
      <div class="pinned-list">
        <div class="small" style="margin-bottom:0.25rem;">Pinned items (watchlist)</div>
        <div id="pinnedList" class="small">Loading...</div>
      </div>
    </div>

    <div class="card">
      <h2>Price history &amp; forecast</h2>
      <div id="priceTitle" class="small">Select an item above to see its price chart.</div>
      <div class="small" style="margin-bottom:0.4rem;">
        <ul style="margin:0; padding-left:1.2rem;">
          <li><strong>Blue</strong>: actual mid price history (up to ~14 days, 5-minute buckets for the last 24 hours, then ~30-minute buckets further back, limited by data availability).</li>
          <li><strong>Green</strong>: current forecast from the latest model, from now into the next 120 minutes.</li>
          <li><strong>Yellow dashed</strong>: forecast that was in effect when you starred the item (if pinned).</li>
        </ul>
      </div>
      <canvas id="priceChart" height="180"></canvas>
      <div id="priceStatus" class="small" style="margin-top:0.4rem;"></div>
    </div>
  </main>

  <script>
  (function () {
    "use strict";

    const statusEl = document.getElementById("status");
    const metaEl = document.getElementById("meta");
    const tableContainer = document.getElementById("tableContainer");
    const priceTitleEl = document.getElementById("priceTitle");
    const priceStatusEl = document.getElementById("priceStatus");
    const chartCanvas = document.getElementById("priceChart");

    const searchInput = document.getElementById("searchInput");
    const searchButton = document.getElementById("searchButton");
    const searchStatusEl = document.getElementById("searchStatus");
    const searchResultsEl = document.getElementById("searchResults");
    const pinnedListEl = document.getElementById("pinnedList");

    const PIN_KEY = "osrs_ge_pins_v3";

    let overviewSignals = [];
    let dailySnapshot = null;
    let mappingList = []; // { id, name }
    let MODEL_HORIZON = 60;
    let MODEL_TAX = 0.02;
    let priceChart = null;

    function loadPinnedState() {
      try {
        const raw = window.localStorage.getItem(PIN_KEY);
        if (!raw) return {};
        const obj = JSON.parse(raw);
        if (obj && typeof obj === "object") return obj;
        return {};
      } catch (e) {
        console.error("Error reading pinned state:", e);
        return {};
      }
    }

    function savePinnedState(state) {
      try {
        window.localStorage.setItem(PIN_KEY, JSON.stringify(state));
      } catch (e) {
        console.error("Error saving pinned state:", e);
      }
    }

    function getPinnedSet() {
      const state = loadPinnedState();
      return new Set(Object.keys(state));
    }

    function buildMappingFromDaily(dailyJson) {
      if (!dailyJson) return;
      const mapping = dailyJson.mapping;
      if (!Array.isArray(mapping)) return;
      mappingList = mapping
        .filter((m) => m && typeof m.id === "number" && m.name)
        .map((m) => ({ id: m.id, name: m.name }));
      mappingList.sort((a, b) => a.name.localeCompare(b.name));
    }

    function formatGp(x) {
      if (x === null || x === undefined || !isFinite(x)) return "-";
      const v = Math.round(x);
      return v.toLocaleString("en-US");
    }

    function formatProfitGp(p) {
      return formatGp(p);
    }

    function formatPercent(p) {
      if (p === null || p === undefined || !isFinite(p)) return "-";
      return (p * 100).toFixed(1) + "%";
    }

    function formatGpPerHour(v) {
      if (v === null || v === undefined || !isFinite(v)) return "-";
      const abs = Math.abs(v);
      if (abs >= 1_000_000_000) return (v / 1_000_000_000).toFixed(1) + "b";
      if (abs >= 1_000_000) return (v / 1_000_000).toFixed(1) + "m";
      if (abs >= 1_000) return (v / 1_000).toFixed(1) + "k";
      return v.toFixed(0);
    }

    function probPill(p) {
      if (p === null || p === undefined || !isFinite(p)) {
        return '<span class="pill">–</span>';
      }
      const pct = p * 100;
      const cls = pct >= 60 ? "pill good" : pct >= 50 ? "pill" : "pill bad";
      return '<span class="' + cls + '">' + pct.toFixed(1) + "%</span>";
    }

    function togglePin(itemId, name, latestForecastPath) {
      const state = loadPinnedState();
      const key = String(itemId);
      if (state[key] && state[key].pinned) {
        delete state[key];
      } else {
        const nowIso = new Date().toISOString();
        state[key] = {
          pinned: true,
          starredAtIso: nowIso,
          pinnedAtIso: nowIso,
          name: name,
          forecastAtStar: Array.isArray(latestForecastPath) ? latestForecastPath : []
        };
      }
      savePinnedState(state);
      renderPinnedList();
      renderTopTable();
    }

    function renderPinnedList() {
      const state = loadPinnedState();
      const entries = Object.entries(state)
        .filter(([, v]) => v && v.pinned)
        .map(([idStr, v]) => ({
          item_id: Number(idStr),
          name: v.name || ("Item " + idStr),
          pinnedAtIso: v.pinnedAtIso || v.starredAtIso || null,
          forecastLen: Array.isArray(v.forecastAtStar) ? v.forecastAtStar.length : 0
        }));

      if (!entries.length) {
        pinnedListEl.textContent = "No pinned items yet.";
        return;
      }

      entries.sort((a, b) => {
        const ta = a.pinnedAtIso ? new Date(a.pinnedAtIso).getTime() : 0;
        const tb = b.pinnedAtIso ? new Date(b.pinnedAtIso).getTime() : 0;
        return tb - ta;
      });

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const trHead = document.createElement("tr");
      ["Item", "Pinned at", "Saved forecast points"].forEach((h) => {
        const th = document.createElement("th");
        th.textContent = h;
        trHead.appendChild(th);
      });
      thead.appendChild(trHead);

      entries.forEach((row) => {
        const tr = document.createElement("tr");
        tr.className = "clickable";
        tr.addEventListener("click", () => {
          loadPriceSeries(row.item_id, row.name);
        });

        const tdName = document.createElement("td");
        tdName.textContent = row.name;
        tr.appendChild(tdName);

        const tdPinned = document.createElement("td");
        tdPinned.textContent = row.pinnedAtIso || "unknown";
        tr.appendChild(tdPinned);

        const tdN = document.createElement("td");
        tdN.textContent = row.forecastLen;
        tr.appendChild(tdN);

        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);
      pinnedListEl.innerHTML = "";
      pinnedListEl.appendChild(table);
    }

    function renderTopTable() {
      if (!overviewSignals.length) {
        tableContainer.textContent = "No signals available.";
        return;
      }

      const vols =
        dailySnapshot &&
        dailySnapshot.volumes_24h &&
        dailySnapshot.volumes_24h.data
          ? dailySnapshot.volumes_24h.data
          : {};

      const pinnedState = loadPinnedState();

      const enriched = overviewSignals
        .filter((s) => s.mid_now && s.mid_now > 0)
        .map((s) => {
          const mid = s.mid_now;
          const grossRet =
            typeof s.future_return_hat === "number" ? s.future_return_hat : 0;

          let netProfit =
            typeof s.expected_profit === "number"
              ? s.expected_profit
              : mid * (1 + grossRet) * (1 - MODEL_TAX) - mid;

          let gpPerSec =
            typeof s.expected_profit_per_second === "number"
              ? s.expected_profit_per_second
              : netProfit / (MODEL_HORIZON * 60);

          const netPct = mid > 0 ? (netProfit / mid) * 100 : 0;

          const keyStr = String(s.item_id);
          const vol24 =
            vols[keyStr] != null
              ? vols[keyStr]
              : vols[s.item_id] != null
              ? vols[s.item_id]
              : null;

          const probProfit =
            typeof s.prob_profit === "number" ? s.prob_profit : 0;

          const volumeWeight =
            vol24 && vol24 > 0
              ? Math.max(0.25, Math.log10(vol24 + 10) / 4)
              : 0.25;

          const probAdjustedProfit =
            Math.max(0, probProfit) * Math.max(0, netProfit);
          const score =
            probAdjustedProfit * Math.max(0.05, probProfit) * volumeWeight;

          const entry = pinnedState[keyStr];
          const isPinned = !!(entry && entry.pinned);

          return {
            item_id: s.item_id,
            name: s.name || "Item " + s.item_id,
            buy_price: mid,
            target_sell_price: mid * (1 + grossRet),
            prob_profit: probProfit,
            netProfit,
            netPct,
            vol24,
            hold_minutes: s.hold_minutes || MODEL_HORIZON,
            pinned: isPinned,
            gpPerSec,
            score,
            rawPath: s.path || []
          };
        });

      const MIN_PROB = 0.55;
      const MIN_NET_PCT = 2.0;

      function compareRows(a, b) {
        if (b.score !== a.score) return b.score - a.score;
        if (b.prob_profit !== a.prob_profit) return b.prob_profit - a.prob_profit;
        if (b.netPct !== a.netPct) return b.netPct - a.netPct;
        return (b.vol24 || 0) - (a.vol24 || 0);
      }

      const good = enriched.filter(
        (row) => row.prob_profit >= MIN_PROB && row.netPct >= MIN_NET_PCT
      );
      const rest = enriched.filter(
        (row) => !(row.prob_profit >= MIN_PROB && row.netPct >= MIN_NET_PCT)
      );

      good.sort(compareRows);
      rest.sort(compareRows);

      const ranked = good.concat(rest);
      const top10 = ranked.slice(0, 10);

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const trHead = document.createElement("tr");
      [
        "★",
        "Item",
        "Buy @",
        "Sell @ (60m)",
        "Profit / item",
        "Net %",
        "GP / sec",
        "Prob.",
        "24h vol",
        "Horizon"
      ].forEach((h) => {
        const th = document.createElement("th");
        th.textContent = h;
        trHead.appendChild(th);
      });
      thead.appendChild(trHead);

      top10.forEach((row) => {
        const tr = document.createElement("tr");
        tr.className = "clickable";

        const tdStar = document.createElement("td");
        const btn = document.createElement("button");
        btn.className = "pin-btn " + (row.pinned ? "pinned" : "unpinned");
        btn.textContent = row.pinned ? "★" : "☆";
        btn.type = "button";
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          togglePin(row.item_id, row.name, row.rawPath);
        });
        tdStar.appendChild(btn);
        tr.appendChild(tdStar);

        const tdName = document.createElement("td");
        tdName.textContent = row.name;
        tr.appendChild(tdName);

        const tdBuy = document.createElement("td");
        tdBuy.textContent = formatGp(row.buy_price);
        tr.appendChild(tdBuy);

        const tdSell = document.createElement("td");
        tdSell.textContent = formatGp(row.target_sell_price);
        tr.appendChild(tdSell);

        const tdProfit = document.createElement("td");
        tdProfit.textContent = formatProfitGp(row.netProfit);
        tr.appendChild(tdProfit);

        const tdNetPct = document.createElement("td");
        tdNetPct.textContent = row.netPct.toFixed(2) + "%";
        tr.appendChild(tdNetPct);

        const tdGpSec = document.createElement("td");
        tdGpSec.textContent =
          row.gpPerSec && isFinite(row.gpPerSec)
            ? formatGp(row.gpPerSec)
            : "-";
        tr.appendChild(tdGpSec);

        const tdProb = document.createElement("td");
        tdProb.innerHTML = probPill(row.prob_profit);
        tr.appendChild(tdProb);

        const tdVol = document.createElement("td");
        tdVol.textContent =
          row.vol24 != null ? row.vol24.toLocaleString("en-US") : "-";
        tr.appendChild(tdVol);

        const tdHoriz = document.createElement("td");
        tdHoriz.textContent = row.hold_minutes + "m";
        tr.appendChild(tdHoriz);

        tr.addEventListener("click", () => {
          loadPriceSeries(row.item_id, row.name);
        });

        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);
      tableContainer.innerHTML = "";
      tableContainer.appendChild(table);

      if (top10.length > 0) {
        loadPriceSeries(top10[0].item_id, top10[0].name);
      }
    }

    function runSearch() {
      const q = (searchInput.value || "").trim().toLowerCase();
      if (!q) {
        searchStatusEl.textContent = "";
        searchResultsEl.innerHTML = "";
        return;
      }
      if (!mappingList.length) {
        searchStatusEl.textContent = "No mapping available yet.";
        searchResultsEl.innerHTML = "";
        return;
      }

      const matches = mappingList.filter((m) =>
        m.name && m.name.toLowerCase().includes(q)
      );
      if (!matches.length) {
        searchStatusEl.textContent = "No items match that query.";
        searchResultsEl.innerHTML = "";
        return;
      }

      searchStatusEl.textContent = "Found " + matches.length + " items.";
      const ul = document.createElement("ul");
      ul.style.listStyle = "none";
      ul.style.paddingLeft = "0";

      matches.slice(0, 25).forEach((m) => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = m.name + " (id " + m.id + ")";
        btn.style.cursor = "pointer";
        btn.style.margin = "0.15rem 0";
        btn.addEventListener("click", () => {
          loadPriceSeries(m.id, m.name);
        });
        li.appendChild(btn);
        ul.appendChild(li);
      });

      searchResultsEl.innerHTML = "";
      searchResultsEl.appendChild(ul);
    }

    searchButton.addEventListener("click", runSearch);
    searchInput.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        runSearch();
      }
    });

    function buildTimeline(history, forecast, starInfo) {
      const tsSet = new Set();
      const histMap = new Map();
      const fcMap = new Map();
      const oldFcMap = new Map();

      function extractTs(pt) {
        if (!pt) return null;
        const iso = pt.timestamp_iso || pt.timestamp || pt.time_iso || pt.time || null;
        if (!iso) return null;
        const ts = Date.parse(iso);
        return Number.isFinite(ts) ? ts : null;
      }

      history.forEach(function (pt) {
        const ts = extractTs(pt);
        if (ts == null) return;
        tsSet.add(ts);
        histMap.set(ts, pt.price);
      });

      forecast.forEach(function (pt) {
        const ts = extractTs(pt);
        if (ts == null) return;
        tsSet.add(ts);
        fcMap.set(ts, pt.price);
      });

      const starForecast =
        starInfo && Array.isArray(starInfo.forecastAtStar)
          ? starInfo.forecastAtStar
          : [];
      starForecast.forEach(function (pt) {
        const ts = extractTs(pt);
        if (ts == null) return;
        tsSet.add(ts);
        oldFcMap.set(ts, pt.price);
      });

      let nowTs = null;
      if (history.length) {
        const ts = extractTs(history[history.length - 1]);
        if (ts != null) nowTs = ts;
      } else if (forecast.length) {
        const ts = extractTs(forecast[0]);
        if (ts != null) nowTs = ts;
      }

      const rawStarTs =
        starInfo && starInfo.starredAtIso
          ? Date.parse(starInfo.starredAtIso)
          : null;
      let starTs = rawStarTs;

      if (starTs != null && histMap.size && !histMap.has(starTs)) {
        const histTsList = Array.from(histMap.keys());
        let closestTs = histTsList[0];
        let smallestDiff = Math.abs(starTs - closestTs);
        for (const ts of histTsList) {
          const diff = Math.abs(starTs - ts);
          if (diff < smallestDiff) {
            smallestDiff = diff;
            closestTs = ts;
          }
        }
        starTs = closestTs;
      }

      if (starTs != null) {
        tsSet.add(starTs);
        if (!oldFcMap.has(starTs) && histMap.has(starTs)) {
          oldFcMap.set(starTs, histMap.get(starTs));
        }
      }
      if (nowTs != null) {
        tsSet.add(nowTs);
      }

      const tsList = Array.from(tsSet).filter(function (ts) {
        return ts != null && Number.isFinite(ts);
      });
      tsList.sort(function (a, b) {
        return a - b;
      });

      const labels = [];
      const histData = [];
      const fcData = [];
      const oldFcData = [];
      const starMarkerData = [];
      const nowMarkerData = [];

      tsList.forEach(function (ts) {
        const d = new Date(ts);
        const iso = isNaN(d.getTime()) ? null : d.toISOString();
        const label = iso ? iso.slice(5, 10) + " " + iso.slice(11, 16) : String(ts);
        labels.push(label);

        histData.push(histMap.has(ts) ? histMap.get(ts) : null);

        const isFutureOrNow = nowTs == null || ts >= nowTs;
        fcData.push(isFutureOrNow && fcMap.has(ts) ? fcMap.get(ts) : null);

        const isAfterStar = starTs == null || ts >= starTs;
        oldFcData.push(isAfterStar && oldFcMap.has(ts) ? oldFcMap.get(ts) : null);

        starMarkerData.push(
          starTs != null && ts === starTs
            ? histMap.get(ts) || oldFcMap.get(ts) || null
            : null
        );
        nowMarkerData.push(
          nowTs != null && ts === nowTs ? histMap.get(ts) || null : null
        );
      });

      return {
        labels,
        histData,
        fcData,
        oldFcData,
        starMarkerData,
        nowMarkerData
      };
    }

    async function loadPriceSeries(itemId, name) {
      priceTitleEl.textContent = "Price for " + name + " (id " + itemId + ")";
      priceStatusEl.textContent = "Loading price series...";

      try {
        const res = await fetch("/price-series?item_id=" + encodeURIComponent(itemId));
        if (!res.ok) {
          priceStatusEl.textContent =
            "No price data available (HTTP " + res.status + ").";
          if (priceChart) {
            priceChart.destroy();
            priceChart = null;
          }
          return;
        }

        const data = await res.json();
        const history = data.history || [];
        const forecast = data.forecast || [];

        if (!history.length && !forecast.length) {
          priceStatusEl.textContent = "No price data yet for this item.";
          if (priceChart) {
            priceChart.destroy();
            priceChart = null;
          }
          return;
        }

        const pinnedState = loadPinnedState();
        const pinEntry = pinnedState[String(itemId)];
        const starInfo =
          pinEntry && pinEntry.pinned
            ? {
                starredAtIso: pinEntry.starredAtIso || pinEntry.pinnedAtIso,
                forecastAtStar: pinEntry.forecastAtStar || []
              }
            : null;

        const tl = buildTimeline(history, forecast, starInfo);
        const labels = tl.labels;
        const histData = tl.histData;
        const fcData = tl.fcData;
        const oldFcData = tl.oldFcData;
        const starMarkerData = tl.starMarkerData;
        const nowMarkerData = tl.nowMarkerData;

        const allPrices = []
          .concat(histData, fcData)
          .filter((v) => v != null && isFinite(v));
        let yMin = 0;
        let yMax = 1;
        if (allPrices.length) {
          const rawMin = Math.min.apply(null, allPrices);
          const rawMax = Math.max.apply(null, allPrices);
          const pad = (rawMax - rawMin) * 0.1 || rawMin * 0.1 || 1;
          yMin = Math.max(0, rawMin - pad);
          yMax = rawMax + pad;
        }

        const ctx = chartCanvas.getContext("2d");
        if (priceChart) {
          priceChart.destroy();
        }

        const datasets = [
          {
            label: "Historical mid price (5m last 24h, 30m older)",
            data: histData,
            borderColor: "rgba(59,130,246,1)",
            backgroundColor: "rgba(59,130,246,0.2)",
            pointRadius: 0,
            borderWidth: 2,
            tension: 0.15,
            spanGaps: true
          },
          {
            label: "Forecast price (next 2h, 5m steps)",
            data: fcData,
            borderColor: "rgba(16,185,129,1)",
            backgroundColor: "rgba(16,185,129,0.15)",
            pointRadius: 0,
            borderWidth: 2,
            borderDash: [6, 3],
            tension: 0.15,
            spanGaps: true
          }
        ];

        if (starInfo && oldFcData.some((v) => v != null)) {
          datasets.splice(1, 0, {
            label: "Forecast at pin time",
            data: oldFcData,
            borderColor: "rgba(234,179,8,1)",
            backgroundColor: "rgba(234,179,8,0.15)",
            pointRadius: 0,
            borderWidth: 1.5,
            borderDash: [6, 4],
            tension: 0.15,
            spanGaps: true
          });
        }

        if (starInfo && starMarkerData.some((v) => v != null)) {
          datasets.push({
            label: "Pin time",
            data: starMarkerData,
            borderColor: "rgba(250,204,21,1)",
            backgroundColor: "rgba(250,204,21,1)",
            pointRadius: 4,
            borderWidth: 0,
            showLine: false
          });
        }

        if (nowMarkerData.some((v) => v != null)) {
          datasets.push({
            label: "Now",
            data: nowMarkerData,
            borderColor: "rgba(248,250,252,1)",
            backgroundColor: "rgba(248,250,252,1)",
            pointRadius: 4,
            borderWidth: 0,
            showLine: false
          });
        }

        priceChart = new Chart(ctx, {
          type: "line",
          data: {
            labels,
            datasets
          },
          options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                ticks: {
                  maxRotation: 45,
                  minRotation: 45,
                  maxTicksLimit: 16
                }
              },
              y: {
                beginAtZero: false,
                suggestedMin: yMin,
                suggestedMax: yMax,
                ticks: {
                  callback: function (value) {
                    const v = Number(value) || 0;
                    if (v >= 1_000_000_000) return (v / 1_000_000_000).toFixed(1) + "b";
                    if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + "m";
                    if (v >= 1_000) return (v / 1_000).toFixed(1) + "k";
                    return v.toFixed(0);
                  }
                }
              }
            },
            plugins: {
              legend: {
                position: "bottom"
              }
            }
          }
        });

        const src =
          data.meta && data.meta.source
            ? data.meta.source
            : "precomputed";
        const truncated =
          data.meta && typeof data.meta.truncated === "boolean"
            ? data.meta.truncated
            : false;

        const hasForecast = forecast && forecast.length > 1;

        if (!hasForecast) {
          priceStatusEl.textContent =
            "No ML forecast for this item (no entry in the latest /signals snapshot). Showing history only. History source: " +
            src +
            (truncated ? " (truncated)" : "") +
            ".";
        } else if (starInfo) {
          priceStatusEl.textContent =
            "History source: " +
            src +
            (truncated ? " (truncated)" : "") +
            ". Blue = history; green = current forecast; yellow dashed = forecast at pin time.";
        } else {
          priceStatusEl.textContent =
            "History source: " +
            src +
            (truncated ? " (truncated)" : "") +
            ". Blue = history; green = current forecast (5–120 minute horizons).";
        }
      } catch (err) {
        console.error("Error loading price series:", err);
        priceStatusEl.textContent = "Error loading price series.";
        if (priceChart) {
          priceChart.destroy();
          priceChart = null;
        }
      }
    }

    async function loadOverview() {
      try {
        statusEl.textContent = "Fetching /signals and /daily...";

        const [sigRes, dailyRes] = await Promise.all([
          fetch("/signals"),
          fetch("/daily")
        ]);

        if (!sigRes.ok) {
          statusEl.textContent =
            "Failed to load signals (HTTP " + sigRes.status + ").";
          return;
        }
        if (!dailyRes.ok) {
          statusEl.textContent =
            "Failed to load daily snapshot (HTTP " + dailyRes.status + ").";
          return;
        }

        const sigJson = await sigRes.json();
        const dailyJson = await dailyRes.json();

        overviewSignals = sigJson.signals || [];
        dailySnapshot = dailyJson;

        MODEL_HORIZON =
          typeof sigJson.horizon_minutes === "number"
            ? sigJson.horizon_minutes
            : 60;
        MODEL_TAX =
          typeof sigJson.tax_rate === "number" ? sigJson.tax_rate : 0.02;

        statusEl.textContent = "Loaded " + overviewSignals.length + " signals.";
        metaEl.textContent =
          "Main horizon: " +
          MODEL_HORIZON +
          " minutes, tax: " +
          (MODEL_TAX * 100).toFixed(1) +
          "%.";

        buildMappingFromDaily(dailySnapshot);
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

//
// Backend (Cloudflare Worker) – R2-backed endpoints
//

// Signals cache
const SIGNALS_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
let LAST_SIGNALS_JSON = null;
let LAST_SIGNALS_FETCHED_AT = 0;

// Simple per-item /price-series response cache
const PRICE_CACHE = new Map();
const PRICE_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const PRICE_CACHE_MAX_ITEMS = 64;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function bucketGetWithRetry(env, key, { attempts = 3, baseDelayMs = 200 } = {}) {
  let lastError = null;
  for (let i = 0; i < attempts; i++) {
    try {
      return await env.OSRS_BUCKET.get(key);
    } catch (err) {
      lastError = err;
      const isLast = i === attempts - 1;
      const delay = baseDelayMs * Math.pow(2, i) + Math.random() * 100;
      console.warn(
        `Attempt ${i + 1} to fetch ${key} failed: ${err.message}${
          isLast ? "" : `; retrying in ${delay}ms`
        }`
      );
      if (!isLast) {
        await sleep(delay);
      }
    }
  }
  console.error(`Failed to fetch ${key} after ${attempts} attempts`, lastError);
  return null;
}

async function loadSignalsWithCache(env) {
  const now = Date.now();
  if (LAST_SIGNALS_JSON && now - LAST_SIGNALS_FETCHED_AT < SIGNALS_CACHE_TTL_MS) {
    return { json: LAST_SIGNALS_JSON, source: "cache" };
  }

  const obj = await bucketGetWithRetry(env, "signals/latest.json");
  if (obj) {
    try {
      const parsed = await obj.json();
      if (parsed && Array.isArray(parsed.signals)) {
        LAST_SIGNALS_JSON = parsed;
        LAST_SIGNALS_FETCHED_AT = Date.now();
        return { json: parsed, source: "fresh" };
      }
    } catch (err) {
      console.error("Error parsing signals/latest.json:", err);
    }
  }

  return { json: LAST_SIGNALS_JSON, source: LAST_SIGNALS_JSON ? "stale" : "missing" };
}

function getCachedPriceSeries(cacheKey) {
  const entry = PRICE_CACHE.get(cacheKey);
  if (!entry) return null;
  const age = Date.now() - entry.cachedAt;
  if (age > PRICE_CACHE_TTL_MS) {
    PRICE_CACHE.delete(cacheKey);
    return null;
  }
  return entry.payload;
}

function setCachedPriceSeries(cacheKey, payload) {
  PRICE_CACHE.set(cacheKey, { cachedAt: Date.now(), payload });
  if (PRICE_CACHE.size > PRICE_CACHE_MAX_ITEMS) {
    const oldestKey = PRICE_CACHE.keys().next().value;
    PRICE_CACHE.delete(oldestKey);
  }
}

async function handleSignals(env) {
  const obj = await bucketGetWithRetry(env, "signals/latest.json");
  if (!obj) {
    return new Response(JSON.stringify({ error: "No signals found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" }
    });
  }
  const text = await obj.text();
  return new Response(text, {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

async function handleDaily(env) {
  const obj = await bucketGetWithRetry(env, "daily/latest.json");
  if (!obj) {
    return new Response(JSON.stringify({ error: "No daily snapshot found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" }
    });
  }
  const text = await obj.text();
  return new Response(text, {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

// Load precomputed per-item history from R2: history/{itemId}.json
async function loadPrecomputedHistory(env, itemId) {
  const key = `history/${itemId}.json`;
  const obj = await bucketGetWithRetry(env, key, { attempts: 2, baseDelayMs: 150 });
  if (!obj) {
    return { history: [], found: false };
  }
  try {
    const parsed = await obj.json();
    if (!parsed || !Array.isArray(parsed.history)) {
      return { history: [], found: false };
    }
    return { history: parsed.history, found: true };
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
  const s = signals.find(
    (row) => row.item_id === itemId || String(row.item_id) === String(itemId)
  );
  if (!s || !Array.isArray(s.path) || typeof s.mid_now !== "number") {
    return forecast;
  }

  const anchorMid = history.length ? history[history.length - 1].price : s.mid_now;
  const baseTimeMs = history.length
    ? Date.parse(history[history.length - 1].timestamp_iso || history[history.length - 1].timestamp)
    : Date.now();

  if (!Number.isFinite(baseTimeMs)) {
    return forecast;
  }

  forecast.push({
    timestamp_iso: new Date(baseTimeMs).toISOString(),
    price: anchorMid
  });

  for (const p of s.path) {
    if (!p || typeof p.minutes !== "number" || typeof p.future_return_hat !== "number") {
      continue;
    }
    const minutes = p.minutes;
    const ret = p.future_return_hat;
    const clampedRet = Math.max(-0.8, Math.min(3.0, ret));
    const price = anchorMid * (1 + clampedRet);
    const tsIso = new Date(baseTimeMs + minutes * 60000).toISOString();
    forecast.push({ timestamp_iso: tsIso, price });
  }

  return forecast;
}

async function handlePriceSeries(env, itemId) {
  const cacheKey = String(itemId);
  const cached = getCachedPriceSeries(cacheKey);
  if (cached) {
    return new Response(cached, {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  }

  const { history, found } = await loadPrecomputedHistory(env, itemId);
  const forecast = await buildForecast(env, itemId, history);

  const truncated = !found;

  const body = JSON.stringify({
    item_id: itemId,
    history,
    forecast,
    meta: {
      truncated,
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

    if (url.pathname === "/signals") {
      return handleSignals(env);
    }
    if (url.pathname === "/daily") {
      return handleDaily(env);
    }
    if (url.pathname === "/price-series") {
      const idParam = url.searchParams.get("item_id");
      const itemId = idParam ? Number(idParam) : NaN;
      if (!Number.isFinite(itemId) || itemId <= 0) {
        return new Response(JSON.stringify({ error: "Invalid item_id" }), {
          status: 400,
          headers: { "Content-Type": "application/json" }
        });
      }
      return handlePriceSeries(env, itemId);
    }

    // default: serve UI
    return new Response(HTML, {
      status: 200,
      headers: { "Content-Type": "text/html; charset=utf-8" }
    });
  }
};
