// OSRS GE UI worker: top trades, search, and price history + forecast (no baseline line)

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
    .pill-watch {
      display: inline-block;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      font-size: 0.75rem;
      background: #111827;
      color: #e5e7eb;
      border: 1px solid #374151;
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
    .pinned-list {
      margin-top: 0.4rem;
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
        Click a row to see up to the last 14 days (or as far back as available) of 5-minute mid prices and a 5–120 minute forecast.<br/>
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
          <li><strong>Blue</strong>: actual mid price history (up to ~14 days, 5-minute buckets, limited by data availability).</li>
          <li><strong>Green</strong>: current forecast from the latest model, from now into the next 120 minutes.</li>
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

    function formatGp(x) {
      if (x === null || x === undefined || !isFinite(x)) return "-";
      const v = Math.round(x);
      return v.toLocaleString("en-US");
    }

    function formatPct(x) {
      if (x === null || x === undefined || !isFinite(x)) return "-";
      return x.toFixed(2) + "%";
    }

    function formatProb(p) {
      if (p === null || p === undefined || !isFinite(p)) return "-";
      return (p * 100).toFixed(1) + "%";
    }

    function probPill(p) {
      if (p === null || p === undefined || !isFinite(p)) {
        return '<span class="pill">n/a</span>';
      }
      let cls = "";
      let label = "";
      if (p >= 0.65) {
        cls = "good";
        label = "favourable";
      } else if (p < 0.5) {
        cls = "bad";
        label = "unfavourable";
      } else {
        label = "uncertain";
      }
      return '<span class="pill ' + cls + '">' +
        label + " (" + formatProb(p) + ")" + "</span>";
    }

    function findMappingList(obj) {
      if (!obj) return null;
      if (Array.isArray(obj)) {
        if (
          obj.length > 0 &&
          typeof obj[0] === "object" &&
          obj[0] !== null &&
          "id" in obj[0] &&
          "name" in obj[0]
        ) {
          return obj;
        }
        for (let i = 0; i < obj.length; i++) {
          const res = findMappingList(obj[i]);
          if (res) return res;
        }
        return null;
      }
      if (typeof obj === "object") {
        for (const k in obj) {
          if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
          const res = findMappingList(obj[k]);
          if (res) return res;
        }
      }
      return null;
    }

    function buildMappingFromDaily() {
      mappingList = [];
      if (!dailySnapshot) return;
      const list = findMappingList(dailySnapshot);
      if (!list) return;
      const out = [];
      for (let i = 0; i < list.length; i++) {
        const entry = list[i];
        if (!entry) continue;
        const idVal = Number(entry.id);
        const nameVal = entry.name != null ? String(entry.name) : "";
        if (!isNaN(idVal) && nameVal) {
          out.push({ id: idVal, name: nameVal });
        }
      }
      mappingList = out;
    }

    async function snapshotForecastAtStar(itemId) {
      try {
        const res = await fetch(
          "/price-series?item_id=" + encodeURIComponent(itemId)
        );
        if (!res.ok) return null;
        const data = await res.json();
        const history = Array.isArray(data.history) ? data.history : [];
        const forecast = Array.isArray(data.forecast) ? data.forecast : [];

        const starTimeIso = history.length
          ? history[history.length - 1].timestamp_iso
          : new Date().toISOString();

        return {
          starredAtIso: starTimeIso,
          forecastAtStar: forecast
        };
      } catch (err) {
        console.error("Failed to snapshot forecast on pin:", err);
        return null;
      }
    }

    async function togglePin(itemId, name, buttonEl) {
      const state = loadPinnedState();
      const key = String(itemId);
      let entry = state[key];
      const wasPinned = !!(entry && entry.pinned);

      if (!entry) {
        entry = {
          name: name,
          pinned: true,
          pinnedAtIso: new Date().toISOString(),
          starredAtIso: null,
          forecastAtStar: []
        };
      } else {
        entry.pinned = !entry.pinned;
        entry.name = name || entry.name;
        if (entry.pinned && !entry.pinnedAtIso) {
          entry.pinnedAtIso = new Date().toISOString();
        }
      }

      if (!wasPinned && entry.pinned) {
        const snap = await snapshotForecastAtStar(itemId);
        if (snap) {
          entry.starredAtIso = snap.starredAtIso;
          entry.forecastAtStar = snap.forecastAtStar;
        } else if (!entry.starredAtIso) {
          entry.starredAtIso = entry.pinnedAtIso;
        }
      }

      state[key] = entry;
      savePinnedState(state);

      if (buttonEl) {
        buttonEl.textContent = entry.pinned ? "★" : "☆";
        buttonEl.classList.toggle("pinned", entry.pinned);
        buttonEl.classList.toggle("unpinned", !entry.pinned);
      }

      renderPinnedList();
    }

    function renderPinnedList() {
      const state = loadPinnedState();
      const items = [];
      for (const key in state) {
        if (!Object.prototype.hasOwnProperty.call(state, key)) continue;
        const entry = state[key];
        if (entry && entry.pinned) {
          items.push({
            item_id: Number(key),
            name: entry.name || ("Item " + key),
            pinnedAtIso: entry.pinnedAtIso || ""
          });
        }
      }

      if (!items.length) {
        pinnedListEl.textContent =
          "No pinned items yet. Click ★ in the table or search results to pin.";
        return;
      }

      items.sort(function (a, b) {
        return a.name.localeCompare(b.name);
      });

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const headerRow = document.createElement("tr");
      ["★", "Item", "Pinned at"].forEach(function (txt) {
        const th = document.createElement("th");
        th.textContent = txt;
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);

      items.forEach(function (row) {
        const tr = document.createElement("tr");
        tr.className = "clickable";

        const tdPin = document.createElement("td");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pin-btn pinned";
        btn.textContent = "★";
        btn.addEventListener("click", async function (ev) {
          ev.stopPropagation();
          await togglePin(row.item_id, row.name, btn);
        });
        tdPin.appendChild(btn);
        tr.appendChild(tdPin);

        const tdName = document.createElement("td");
        tdName.textContent = row.name + " ";
        const spanId = document.createElement("span");
        spanId.className = "small mono";
        spanId.textContent = "(id " + row.item_id + ")";
        tdName.appendChild(spanId);
        tr.appendChild(tdName);

        const tdTime = document.createElement("td");
        tdTime.className = "small mono";
        tdTime.textContent = row.pinnedAtIso || "-";
        tr.appendChild(tdTime);

        tr.addEventListener("click", function () {
          loadPriceSeries(row.item_id, row.name);
        });

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
        .filter(function (s) {
          return s.mid_now && s.mid_now > 0;
        })
        .map(function (s) {
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
          const grossPct = grossRet * 100;

          const keyStr = String(s.item_id);
          const vol24 =
            vols[keyStr] != null
              ? vols[keyStr]
              : vols[s.item_id] != null
              ? vols[s.item_id]
              : null;

          const probProfit =
            typeof s.prob_profit === "number" ? s.prob_profit : 0;

          // Give liquid items a modest boost while still letting low-volume picks compete.
          const volumeWeight = vol24 && vol24 > 0
            ? Math.max(0.25, Math.log10(vol24 + 10) / 4)
            : 0.25;
          const probAdjustedProfit =
            Math.max(0, probProfit) * Math.max(0, netProfit);
          // Keep a floor on probability to avoid a zeroed score for very small but non-zero odds.
          const score =
            probAdjustedProfit * Math.max(0.05, probProfit) * volumeWeight;

          const entry = pinnedState[keyStr];
          const isPinned = !!(entry && entry.pinned);

          return {
            item_id: s.item_id,
            name: s.name || "Item " + s.item_id,
            buy_price: mid,
            target_sell_price: mid * (1 + grossRet),
            mid_now: mid,
            prob_profit: probProfit,
            netProfit: netProfit,
            netPct: netPct,
            grossPct: grossPct,
            vol24: vol24,
            hold_minutes: s.hold_minutes || MODEL_HORIZON,
            pinned: isPinned,
            gpPerSec: gpPerSec,
            score: score
          };
        });

      // Thresholds for "good" trades
      const MIN_PROB = 0.55;   // at least ~55% win chance
      const MIN_NET_PCT = 2.0; // at least 2% net return per item

      function compareRows(a, b) {
        if (b.score !== a.score) return b.score - a.score;
        if (b.prob_profit !== a.prob_profit) return b.prob_profit - a.prob_profit;
        if (b.netPct !== a.netPct) return b.netPct - a.netPct;
        return (b.vol24 || 0) - (a.vol24 || 0);
      }

      const good = enriched.filter(function (row) {
        return row.prob_profit >= MIN_PROB && row.netPct >= MIN_NET_PCT;
      });

      const rest = enriched.filter(function (row) {
        return !(row.prob_profit >= MIN_PROB && row.netPct >= MIN_NET_PCT);
      });

      good.sort(compareRows);
      rest.sort(compareRows);

      const ranked = good.concat(rest);
      const top10 = ranked.slice(0, 10);

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const headerRow = document.createElement("tr");
      const headers = [
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
      ];
      headers.forEach(function (text) {
        const th = document.createElement("th");
        th.textContent = text;
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);

      top10.forEach(function (row) {
        const tr = document.createElement("tr");
        tr.className = "clickable";
        tr.dataset.itemId = String(row.item_id);
        tr.dataset.name = row.name;

        const tdPin = document.createElement("td");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pin-btn " + (row.pinned ? "pinned" : "unpinned");
        btn.textContent = row.pinned ? "★" : "☆";
        btn.addEventListener("click", async function (ev) {
          ev.stopPropagation();
          await togglePin(row.item_id, row.name, btn);
        });
        tdPin.appendChild(btn);
        tr.appendChild(tdPin);

        const tdName = document.createElement("td");
        tdName.textContent = row.name + " ";
        const spanId = document.createElement("span");
        spanId.className = "small mono";
        spanId.textContent = "(id " + row.item_id + ")";
        tdName.appendChild(spanId);
        tr.appendChild(tdName);

        const tdBuy = document.createElement("td");
        tdBuy.className = "mono";
        tdBuy.textContent = formatGp(row.buy_price);
        tr.appendChild(tdBuy);

        const tdSell = document.createElement("td");
        tdSell.className = "mono";
        tdSell.textContent = formatGp(row.target_sell_price);
        tr.appendChild(tdSell);

        const tdProfit = document.createElement("td");
        tdProfit.className = "mono";
        tdProfit.textContent = formatGp(row.netProfit);
        tr.appendChild(tdProfit);

        const tdNetPct = document.createElement("td");
        tdNetPct.textContent = formatPct(row.netPct);
        tr.appendChild(tdNetPct);

        const tdGps = document.createElement("td");
        if (row.gpPerSec != null && isFinite(row.gpPerSec)) {
          tdGps.textContent = row.gpPerSec.toFixed(4);
        } else {
          tdGps.textContent = "-";
        }
        tr.appendChild(tdGps);

        const tdProb = document.createElement("td");
        tdProb.innerHTML = probPill(row.prob_profit);
        tr.appendChild(tdProb);

        const tdVol = document.createElement("td");
        if (row.vol24 != null) {
          tdVol.textContent = row.vol24.toLocaleString("en-US");
        } else {
          tdVol.textContent = "-";
        }
        tr.appendChild(tdVol);

        const tdH = document.createElement("td");
        tdH.textContent = row.hold_minutes + "m";
        tr.appendChild(tdH);

        tr.addEventListener("click", function () {
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
      if (!mappingList.length) {
        searchStatusEl.textContent =
          "No item mapping available yet (daily snapshot is missing mapping list).";
        searchResultsEl.innerHTML = "";
        return;
      }

      const qRaw = searchInput.value.trim();
      if (!qRaw) {
        searchStatusEl.textContent = "Type an item name to search.";
        searchResultsEl.innerHTML = "";
        return;
      }
      const q = qRaw.toLowerCase();

      const results = mappingList
        .filter(function (entry) {
          return entry.name.toLowerCase().indexOf(q) !== -1;
        })
        .slice(0, 25);

      if (!results.length) {
        searchStatusEl.textContent =
          'No items found for "' + qRaw + '".';
        searchResultsEl.innerHTML = "";
        return;
      }

      searchStatusEl.textContent =
        "Found " + results.length + " item(s). Showing up to 25.";

      const state = loadPinnedState();
      const vols =
        dailySnapshot &&
        dailySnapshot.volumes_24h &&
        dailySnapshot.volumes_24h.data
          ? dailySnapshot.volumes_24h.data
          : {};

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const headerRow = document.createElement("tr");
      ["★", "Item", "24h vol"].forEach(function (txt) {
        const th = document.createElement("th");
        th.textContent = txt;
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);

      results.forEach(function (res) {
        const id = res.id;
        const name = res.name;
        const keyStr = String(id);
        const entry = state[keyStr];
        const isPinned = entry && entry.pinned;

        const tr = document.createElement("tr");
        tr.className = "clickable";

        const tdPin = document.createElement("td");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pin-btn " + (isPinned ? "pinned" : "unpinned");
        btn.textContent = isPinned ? "★" : "☆";
        btn.addEventListener("click", async function (ev) {
          ev.stopPropagation();
          await togglePin(id, name, btn);
        });
        tdPin.appendChild(btn);
        tr.appendChild(tdPin);

        const tdName = document.createElement("td");
        tdName.textContent = name + " ";
        const spanId = document.createElement("span");
        spanId.className = "small mono";
        spanId.textContent = "(id " + id + ")";
        tdName.appendChild(spanId);
        tr.appendChild(tdName);

        const tdVol = document.createElement("td");
        const v =
          vols[keyStr] != null
            ? vols[keyStr]
            : vols[id] != null
            ? vols[id]
            : null;
        tdVol.textContent = v != null ? v.toLocaleString("en-US") : "-";
        tr.appendChild(tdVol);

        tr.addEventListener("click", function () {
          loadPriceSeries(id, name);
        });

        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);
      searchResultsEl.innerHTML = "";
      searchResultsEl.appendChild(table);
    }

    function buildTimeline(history, forecast, starInfo) {
      const tsSet = new Set();
      const histMap = new Map();
      const fcMap = new Map();
      const oldFcMap = new Map();

      history.forEach(function (pt) {
        const ts = new Date(pt.timestamp_iso).getTime();
        tsSet.add(ts);
        histMap.set(ts, pt.price);
      });

      forecast.forEach(function (pt) {
        const ts = new Date(pt.timestamp_iso).getTime();
        tsSet.add(ts);
        fcMap.set(ts, pt.price);
      });

      const starForecast =
        starInfo && Array.isArray(starInfo.forecastAtStar)
          ? starInfo.forecastAtStar
          : [];
      starForecast.forEach(function (pt) {
        const ts = new Date(pt.timestamp_iso).getTime();
        tsSet.add(ts);
        oldFcMap.set(ts, pt.price);
      });

      const nowTs = history.length
        ? new Date(history[history.length - 1].timestamp_iso).getTime()
        : forecast.length
        ? new Date(forecast[0].timestamp_iso).getTime()
        : null;

      const rawStarTs = starInfo && starInfo.starredAtIso
        ? new Date(starInfo.starredAtIso).getTime()
        : null;
      const starTsCandidate = rawStarTs;
      let starTs = starTsCandidate;
      // Snap the star timestamp to the nearest history bucket so the overlay connects cleanly.
      if (starTsCandidate != null && histMap.size && !histMap.has(starTsCandidate)) {
        const lastHistTs = new Date(
          history[history.length - 1].timestamp_iso
        ).getTime();
        let closestTs = lastHistTs;
        let smallestDiff = Math.abs(starTsCandidate - lastHistTs);
        histMap.forEach(function (_price, ts) {
          const diff = Math.abs(starTsCandidate - ts);
          if (diff < smallestDiff) {
            smallestDiff = diff;
            closestTs = ts;
          }
        });
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

      const tsList = Array.from(tsSet);
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
        const iso = d.toISOString();
        const label = iso.slice(5, 10) + " " + iso.slice(11, 16); // MM-DD HH:MM
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
        labels: labels,
        histData: histData,
        fcData: fcData,
        oldFcData: oldFcData,
        starMarkerData: starMarkerData,
        nowMarkerData: nowMarkerData
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
          if (priceChart) { priceChart.destroy(); priceChart = null; }
          return;
        }

        const data = await res.json();
        const history = data.history || [];
        const forecast = data.forecast || [];

        if (!history.length && !forecast.length) {
          priceStatusEl.textContent = "No price data yet for this item.";
          if (priceChart) { priceChart.destroy(); priceChart = null; }
          return;
        }

        const pinnedState = loadPinnedState();
        const pinEntry = pinnedState[String(itemId)];
        const starInfo = pinEntry && pinEntry.pinned
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
          .filter(function (v) { return v != null && isFinite(v); });
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
            label: "Historical mid price (5m)",
            data: histData,
            borderColor: "rgba(59,130,246,1)",
            backgroundColor: "rgba(59,130,246,0.2)",
            pointRadius: 0,
            borderWidth: 2,
            tension: 0.15
          },
          {
            label: "Forecast price (next 2h, 5m steps)",
            data: fcData,
            borderColor: "rgba(16,185,129,1)",
            backgroundColor: "rgba(16,185,129,0.15)",
            pointRadius: 0,
            borderWidth: 2,
            borderDash: [6, 3],
            tension: 0.15
          }
        ];

        if (starInfo && oldFcData.some(function (v) { return v != null; })) {
          datasets.splice(1, 0, {
            label: "Forecast at star time",
            data: oldFcData,
            borderColor: "rgba(234,179,8,1)",
            backgroundColor: "rgba(234,179,8,0.1)",
            pointRadius: 0,
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.15
          });
        }

        if (starInfo) {
          datasets.push({
            label: "Current time",
            data: nowMarkerData,
            borderColor: "rgba(248,113,113,1)",
            backgroundColor: "rgba(248,113,113,1)",
            pointRadius: 5,
            borderWidth: 0,
            showLine: false
          });
        }

        if (starInfo && starMarkerData.some(function (v) { return v != null; })) {
          datasets.push({
            label: "Starred at",
            data: starMarkerData,
            borderColor: "rgba(234,179,8,1)",
            backgroundColor: "rgba(234,179,8,1)",
            pointRadius: 5,
            borderWidth: 0,
            showLine: false
          });
        }

        priceChart = new Chart(ctx, {
          type: "line",
          data: {
            labels: labels,
            datasets: datasets
          },
          options: {
            responsive: true,
            scales: {
              y: {
                beginAtZero: false,
                suggestedMin: yMin,
                suggestedMax: yMax,
                title: { display: true, text: "Price (gp)" }
              },
              x: {
                ticks: {
                  maxRotation: 45,
                  minRotation: 45
                }
              }
            },
            plugins: {
              legend: {
                labels: { color: "#e5e7eb" }
              }
            }
          }
        });

        priceStatusEl.textContent = starInfo
          ? "Blue = actual prices; green = current forecast from the latest model (5–120 minute horizons); yellow dashed = forecast captured when you starred the item."
          : "Blue = actual prices; green = current forecast from the latest model (5–120 minute horizons).";
      } catch (err) {
        console.error(err);
        priceStatusEl.textContent = "Error loading price series: " + err.message;
        if (priceChart) { priceChart.destroy(); priceChart = null; }
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
          "Main horizon: " + MODEL_HORIZON + " minutes, tax: " +
          (MODEL_TAX * 100).toFixed(1) + "%.";

        buildMappingFromDaily();
        renderTopTable();
        renderPinnedList();
      } catch (err) {
        console.error(err);
        statusEl.textContent = "Error loading overview: " + err.message;
        tableContainer.textContent = "Failed to load.";
      }
    }

    searchButton.addEventListener("click", runSearch);
    searchInput.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        runSearch();
      }
    });

    loadOverview();
  })();
  </script>
</body>
</html>`;

// ----------------- Backend helpers -----------------

const PRICE_CACHE = new Map();
const PRICE_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const PRICE_CACHE_MAX_ITEMS = 50;

function getCachedPriceSeries(cacheKey, { allowStale = false } = {}) {
  const entry = PRICE_CACHE.get(cacheKey);
  if (!entry) return null;

  const age = Date.now() - entry.cachedAt;
  if (age > PRICE_CACHE_TTL_MS && !allowStale) {
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
  const obj = await env.OSRS_BUCKET.get("signals/latest.json");
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
  const obj = await env.OSRS_BUCKET.get("daily/latest.json");
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

// Build price history (~14d of 5m mid prices) + forecast from signals.path,
// anchoring the forecast to the last history mid price.
async function buildSnapshotKeys(env, startedAt, budgetMs) {
  const MAX_DAYS = 14;
  const MAX_SNAPSHOTS = MAX_DAYS * 24 * 12; // 5m buckets
  const now = new Date();

  const keys = [];
  let truncated = false;

  for (let delta = 0; delta < MAX_DAYS && keys.length < MAX_SNAPSHOTS; delta++) {
    if (Date.now() - startedAt > budgetMs) {
      truncated = true;
      break;
    }

    const d = new Date(now.getTime() - delta * 86400000);
    const year = d.getUTCFullYear();
    const month = String(d.getUTCMonth() + 1).padStart(2, "0");
    const day = String(d.getUTCDate()).padStart(2, "0");
    const prefix = "5m/" + year + "/" + month + "/" + day + "/";

    let cursor;
    while (true) {
      const listing = await env.OSRS_BUCKET.list({ prefix, cursor, limit: 1000 });
      for (const obj of listing.objects || []) {
        keys.push(obj.key);
      }
      if (!listing.truncated) break;
      if (Date.now() - startedAt > budgetMs) {
        truncated = true;
        break;
      }
      cursor = listing.cursor;
    }
    if (keys.length >= MAX_SNAPSHOTS || truncated) break;
  }

  keys.sort();
  return { selectedKeys: keys.slice(-MAX_SNAPSHOTS), truncated };
}

function sampleSnapshotKeys(selectedKeys) {
  // Downsample snapshot fetches to avoid timeouts while still covering the full window.
  // Keep the most recent 3h at full (5m) resolution and aggressively sample the older
  // portion to stay within a tighter fetch budget. The lower cap keeps the worker comfortably
  // within observed timeout thresholds and reduces cascading failures when a single item is heavy
  // to load.
  const MAX_FETCH_KEYS = 90; // tighter hard cap on snapshot fetches per request
  const RECENT_FULL_WINDOW = 3 * 12; // last 3h of 5m snapshots

  const recentKeys = selectedKeys.slice(-RECENT_FULL_WINDOW);
  const olderKeys = selectedKeys.slice(0, Math.max(0, selectedKeys.length - RECENT_FULL_WINDOW));

  const remainingBudget = Math.max(0, MAX_FETCH_KEYS - recentKeys.length);
  const sampledOlder = [];
  if (olderKeys.length && remainingBudget > 0) {
    const olderStride = Math.max(1, Math.ceil(olderKeys.length / remainingBudget));
    for (let i = 0; i < olderKeys.length; i += olderStride) {
      sampledOlder.push(olderKeys[i]);
    }
  }

  const sampledKeys = sampledOlder.concat(recentKeys);
  if (sampledKeys.length && sampledKeys[sampledKeys.length - 1] !== selectedKeys[selectedKeys.length - 1]) {
    sampledKeys.push(selectedKeys[selectedKeys.length - 1]);
  }

  return sampledKeys;
}

async function readSnapshotJson(env, key, maxAttempts = 2) {
  let lastError = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const obj = await env.OSRS_BUCKET.get(key).then(
      (res) => res,
      (err) => {
        lastError = err;
        return null;
      }
    );
    if (!obj) return null;

    const snap = await obj.json().then(
      (json) => json,
      (err) => {
        lastError = err;
        return null;
      }
    );
    if (snap) return snap;
  }

  if (lastError) {
    console.error("Failed to read snapshot", key, lastError);
  }
  return null;
}

async function loadHistoryFromSnapshots(env, itemId, sampledKeys, startedAt, budgetMs) {
  const history = [];
  const BATCH_SIZE = 8;
  let truncated = false;

  for (let i = 0; i < sampledKeys.length; i += BATCH_SIZE) {
    if (Date.now() - startedAt > budgetMs) {
      truncated = true;
      break;
    }

    const batch = sampledKeys.slice(i, i + BATCH_SIZE);
    const results = await Promise.all(
      batch.map(async (key) => {
        const snap = await readSnapshotJson(env, key);
        if (!snap) return null;

        const fm = snap && (snap.five_minute || snap["five_minute"]);
        if (!fm || !fm.data) return null;

        const rec = fm.data[String(itemId)] || fm.data[itemId];
        if (!rec) return null;

        const ah = rec.avgHighPrice;
        const al = rec.avgLowPrice;
        if (typeof ah !== "number" || typeof al !== "number" || ah <= 0 || al <= 0) return null;

        const mid = (ah + al) / 2;
        const tsSec = typeof fm.timestamp === "number" ? fm.timestamp : null;
        if (!tsSec) return null;

        return {
          timestamp_iso: new Date(tsSec * 1000).toISOString(),
          price: mid
        };
      })
    );
    for (const entry of results) {
      if (entry) history.push(entry);
    }
  }

  history.sort((a, b) => a.timestamp_iso.localeCompare(b.timestamp_iso));
  return { history, truncated };
}

async function buildForecast(env, itemId, history) {
  const forecast = [];
  const sigObj = await env.OSRS_BUCKET.get("signals/latest.json").catch((err) => {
    console.error("Error fetching signals for forecast:", err);
    return null;
  });
  if (!sigObj) return forecast;

  const sigJson = await sigObj.json().catch((err) => {
    console.error("Error parsing signals for forecast:", err);
    return null;
  });
  if (!sigJson) return forecast;

  const signals = sigJson.signals || [];
  const s = signals.find((row) => row.item_id === itemId || String(row.item_id) === String(itemId));
  if (!s || !Array.isArray(s.path) || typeof s.mid_now !== "number") return forecast;

  const anchorMid = history.length ? history[history.length - 1].price : s.mid_now;

  const baseTimeMs = history.length
    ? new Date(history[history.length - 1].timestamp_iso).getTime()
    : Date.now();

  // Anchor the forecast to the last history timestamp so the lines connect
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

    // Optional mild clamp to avoid absurd spikes
    const clampedRet = Math.max(-0.8, Math.min(3.0, ret));

    const price = anchorMid * (1 + clampedRet);
    const tsIso = new Date(baseTimeMs + minutes * 60000).toISOString();
    forecast.push({
      timestamp_iso: tsIso,
      price: price
    });
  }

  return forecast;
}

async function buildPriceSeriesPayload(env, itemId, startedAt, budgetMs) {
  const { selectedKeys, truncated: listTruncated } = await buildSnapshotKeys(
    env,
    startedAt,
    budgetMs
  );
  let truncated = !!listTruncated;

  const sampledKeys = sampleSnapshotKeys(selectedKeys);
  const historyResult = await loadHistoryFromSnapshots(
    env,
    itemId,
    sampledKeys,
    startedAt,
    budgetMs
  );
  truncated = truncated || historyResult.truncated;
  const history = historyResult.history;

  const forecast = await buildForecast(env, itemId, history);

  return {
    truncated,
    body: JSON.stringify({
      item_id: itemId,
      history,
      forecast,
      meta: { truncated, source: "fresh" }
    })
  };
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

  const stale = getCachedPriceSeries(cacheKey, { allowStale: true });
  const startedAt = Date.now();
  const BUILD_BUDGET_MS = 12_000; // soft budget to bail out before hard worker timeout

  const built = await buildPriceSeriesPayload(env, itemId, startedAt, BUILD_BUDGET_MS).then(
    (payload) => payload,
    (err) => {
      console.error("Error building price series:", err);
      return null;
    }
  );

  if (!built) {
    if (stale) {
      return new Response(stale, {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "X-Price-Series-Stale": "1"
        }
      });
    }

    return new Response(
      JSON.stringify({ error: "Failed to build price series, please retry." }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }

  setCachedPriceSeries(cacheKey, built.body);
  return new Response(built.body, {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      ...(built.truncated ? { "X-Price-Series-Partial": "1" } : {})
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
      const itemId = parseInt(idParam, 10);
      if (!idParam || !Number.isFinite(itemId)) {
        return new Response(
          JSON.stringify({ error: "item_id query param required" }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
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
