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
    .chart-wrapper {
      position: relative;
      height: 260px;
      max-width: 100%;
    }
    .chart-wrapper canvas {
      display: block;
      width: 100% !important;
      height: 100% !important;
    }
    .pagination-controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 0.35rem 0;
      font-size: 0.8rem;
      color: #9ca3af;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .pagination-controls select {
      margin-left: 0.25rem;
      padding: 0.15rem 0.25rem;
      border-radius: 0.35rem;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
    }
    .pagination-controls button {
      padding: 0.25rem 0.6rem;
      border-radius: 0.35rem;
      border: 1px solid #4b5563;
      background: #111827;
      color: #e5e7eb;
      cursor: pointer;
      font-size: 0.8rem;
    }
    .pagination-controls button:hover:not(:disabled) {
      background: #1f2937;
    }
	    .pagination-controls button:disabled {
	      opacity: 0.5;
	      cursor: default;
	    }
	    .peaks-tab-layout {
	      display: flex;
	      gap: 1rem;
	      align-items: flex-start;
	    }
	    .peaks-tab-main {
	      flex: 1;
	      min-width: 0;
	    }
	    .peaks-tab-side {
	      width: 260px;
	    }
	    .peaks-sort-card h3 {
	      margin: 0 0 0.6rem 0;
	      font-size: 0.95rem;
	      color: #e5e7eb;
	    }
	    .peaks-weight-row {
	      display: flex;
	      flex-direction: column;
	      gap: 0.25rem;
	      margin-bottom: 0.7rem;
	    }
	    .peaks-weight-label {
	      display: flex;
	      justify-content: space-between;
	      font-size: 0.8rem;
	      color: #9ca3af;
	    }
	    .peaks-weight-row input[type="range"] {
	      width: 100%;
	    }
	    .peaks-apply-btn {
	      width: 100%;
	      margin-top: 0.25rem;
	      padding: 0.35rem 0.6rem;
	      border-radius: 0.35rem;
	      border: 1px solid #4b5563;
	      background: #111827;
	      color: #e5e7eb;
	      cursor: pointer;
	      font-size: 0.85rem;
	    }
	    .peaks-apply-btn:hover {
	      background: #1f2937;
	    }
	    @media (max-width: 750px) {
	      table { font-size: 0.78rem; }
	      header h1 { font-size: 1.2rem; }
	      .card { padding: 0.85rem 0.9rem; }
	      .chart-wrapper { height: 220px; }
	    }
	    @media (max-width: 900px) {
	      .peaks-tab-layout {
	        flex-direction: column;
	      }
	      .peaks-tab-side {
	        width: 100%;
	      }
	    }
	    .tabs {
	      display: flex;
	      gap: 0.25rem;
	      margin: 0.75rem 0 1rem;
	      border-bottom: 1px solid #1f2937;
    }
    .tab-btn {
      appearance: none;
      border: 1px solid #1f2937;
      border-bottom: none;
      background: #020617;
      color: #9ca3af;
      padding: 0.5rem 0.9rem;
      border-top-left-radius: 0.5rem;
      border-top-right-radius: 0.5rem;
      cursor: pointer;
      font-size: 0.9rem;
    }
    .tab-btn.active {
      color: #e5e7eb;
      background: #0b1220;
      border-color: #374151;
    }
    .tab-content {
      display: none;
    }
    .tab-content.active {
      display: block;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2"></script>
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
    <div class="tabs" role="tablist">
      <button type="button" class="tab-btn active" data-tab="standard" role="tab" aria-selected="true">Standard Model</button>
      <button type="button" class="tab-btn" data-tab="peaks" role="tab" aria-selected="false">Catching Peaks</button>
    </div>

    <div id="tab-standard" class="tab-content active" role="tabpanel" aria-hidden="false">
      <div class="card">
        <div id="status">Loading signals &amp; daily snapshot...</div>
        <div id="meta"></div>
      </div>

      <div class="card">
        <h2>Ranked items by probability-adjusted profit × liquidity</h2>
        <div class="small" style="margin-bottom:0.4rem;">
          Click a row to see up to the last 14 days (or as far back as available) of mid prices:
          5-minute buckets for the last 24 hours, then ~30-minute buckets further back, plus a 5–120 minute forecast.<br/>
          Click the ★ column to pin an item. Pins persist across refreshes.<br/>
          Regime column: H = high-value (≥100k), M = mid-value (10k–100k), L = low-value (&lt;10k, noisier / experimental).
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
          <div id="pinnedList" class="small">No pinned items yet.</div>
        </div>
      </div>

      <div id="standardChartMount"></div>
    </div>

	    <div id="tab-peaks" class="tab-content" role="tabpanel" aria-hidden="true">
	      <div class="peaks-tab-layout">
	        <div class="peaks-tab-main">
	          <div class="card">
	            <div id="peaksStatus">Loading catching‑peaks metrics...</div>
	            <div id="peaksMeta" class="small"></div>
	          </div>

	          <div class="card">
	            <h2>Catching Peaks leaderboard</h2>
	            <div class="small" style="margin-bottom:0.4rem;">
	              Items with a stable low baseline most of the time, punctuated by rare, short-lived high spikes.
	              Computed by fitting a two‑state Gaussian mixture (baseline vs spike) to recent mid prices.
	            </div>
	            <div id="peaksTableContainer">Waiting for data...</div>
	          </div>

	          <div id="peaksChartMount"></div>
	        </div>

	        <div class="peaks-tab-side">
	          <div class="card peaks-sort-card">
	            <h3>Weighted sort</h3>
	            <div id="peaksSortPane"></div>
	          </div>
	        </div>
	      </div>
	    </div>

    <div id="priceCard" class="card">
      <h2>Price history &amp; forecast</h2>
      <div id="priceTitle" class="small">Select an item above to see its price chart.</div>
      <div class="small" style="margin-bottom:0.4rem;">
        <ul style="margin:0; padding-left:1.2rem;">
          <li><strong>Blue</strong>: actual mid price history (up to ~14 days, 5-minute buckets for the last 24 hours, then ~30-minute buckets further back, limited by data availability).</li>
          <li><strong>Green</strong>: current forecast from the latest model, from now into the next 120 minutes.</li>
          <li><strong>Yellow dashed</strong>: forecast that was in effect when you starred the item (if pinned).</li>
        </ul>
      </div>
      <div class="chart-wrapper">
        <canvas id="priceChart"></canvas>
      </div>
      <div id="priceStatus" class="small" style="margin-top:0.4rem;"></div>
      <div class="small" style="margin-top:0.25rem;">Scroll to zoom, Shift+drag to pan, double‑click to reset.</div>
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

	    const peaksStatusEl = document.getElementById("peaksStatus");
	    const peaksMetaEl = document.getElementById("peaksMeta");
	    const peaksTableContainer = document.getElementById("peaksTableContainer");
	    const peaksSortPaneEl = document.getElementById("peaksSortPane");
	    const standardChartMount = document.getElementById("standardChartMount");
	    const peaksChartMount = document.getElementById("peaksChartMount");
	    const priceCardEl = document.getElementById("priceCard");
	    const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
    const tabStandardEl = document.getElementById("tab-standard");
    const tabPeaksEl = document.getElementById("tab-peaks");

    const PIN_KEY = "osrs_ge_pins_v3";
    const ACTIVE_TAB_KEY = "osrs_ge_active_tab_v1";

	    let overviewSignals = [];
	    let dailySnapshot = null;
	    let mappingList = [];
	    let mappingById = new Map();
	    let volumes24hById = new Map();
	    let MODEL_HORIZON = 60;
	    let MODEL_TAX = 0.02;
	    let priceChart = null;
    // Ranking pagination
    let rankingPageSize = 10;
    let rankingCurrentPage = 1;
    const RANKING_PAGE_SIZE_OPTIONS = [10, 50, 100, 250, 500];
	    // Catching Peaks pagination
	    let peaksPageSize = 10;
	    let peaksCurrentPage = 1;
	    const PEAKS_PAGE_SIZE_OPTIONS = [10, 50, 100, 250, 500];
		    let peaksItems = [];
		    let peaksLoaded = false;
		    let peaksSortKey = "sharpness";
		    let peaksSortDir = "desc";
		    const DEFAULT_PEAK_WEIGHT = 100;
		    let peaksPercentWeights = {};
		    let peaksSortPaneKeys = [];
		    // Latest volume timeline for the active chart (aligned to labels)
		    let latestVolumeTimeline = [];

    function moveChartToTab(tabName) {
      const mount =
        tabName === "peaks" ? peaksChartMount : standardChartMount;
      if (mount && priceCardEl && priceCardEl.parentElement !== mount) {
        mount.appendChild(priceCardEl);
      }
    }

    function setActiveTab(tabName) {
      const active = tabName === "peaks" ? "peaks" : "standard";
      tabButtons.forEach((btn) => {
        const isActive = btn.dataset.tab === active;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
      });
      if (tabStandardEl) {
        const showStandard = active === "standard";
        tabStandardEl.classList.toggle("active", showStandard);
        tabStandardEl.setAttribute("aria-hidden", showStandard ? "false" : "true");
      }
      if (tabPeaksEl) {
        const showPeaks = active === "peaks";
        tabPeaksEl.classList.toggle("active", showPeaks);
        tabPeaksEl.setAttribute("aria-hidden", showPeaks ? "false" : "true");
      }
      moveChartToTab(active);
      try {
        window.localStorage.setItem(ACTIVE_TAB_KEY, active);
      } catch (_) {}

      if (active === "peaks" && !peaksLoaded) {
        loadPeaksOverview();
      }
    }

    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        setActiveTab(btn.dataset.tab);
      });
    });

    function loadPinnedState() {
      try {
        const raw = window.localStorage.getItem(PIN_KEY);
        if (!raw) return {};
        const obj = JSON.parse(raw);
        if (obj && typeof obj === "object") return obj;
      } catch (err) {
        console.warn("Failed to parse pin state:", err);
      }
      return {};
    }

    function savePinnedState(state) {
      try {
        window.localStorage.setItem(PIN_KEY, JSON.stringify(state));
      } catch (err) {
        console.warn("Failed to save pin state:", err);
      }
    }

    function getPinnedSet() {
      const state = loadPinnedState();
      return new Set(Object.keys(state));
    }

    function formatProfitGp(p) {
      if (!Number.isFinite(p)) return "-";
      const v = Math.round(p);
      return v.toLocaleString("en-US");
    }

    function formatPercent(prob) {
      if (!Number.isFinite(prob)) return "-";
      return (prob * 100).toFixed(1) + "%";
    }

    function formatGpPerHour(v) {
      if (!Number.isFinite(v)) return "-";
      const abs = Math.abs(v);
      if (abs >= 1_000_000_000) return (v / 1_000_000_000).toFixed(1) + "b";
      if (abs >= 1_000_000) return (v / 1_000_000).toFixed(1) + "m";
      if (abs >= 1_000) return (v / 1_000).toFixed(1) + "k";
      return v.toFixed(0);
    }

    function probPill(p) {
      if (!Number.isFinite(p)) return '<span class="pill">–</span>';
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
          pinnedAtIso: nowIso,
          starredAtIso: nowIso,
          name: name,
          forecastAtStar: Array.isArray(latestForecastPath) ? latestForecastPath : []
        };
      }
      savePinnedState(state);
      renderPinnedList();
      renderTopTable();
    }

    function buildMappingFromDaily(dailyJson) {
      mappingList = [];
      mappingById = new Map();
      if (!dailyJson) return;

      function dfs(obj) {
        if (Array.isArray(obj)) {
          if (
            obj.length &&
            typeof obj[0] === "object" &&
            obj[0] &&
            "id" in obj[0] &&
            "name" in obj[0]
          ) {
            return obj;
          }
          for (const el of obj) {
            const r = dfs(el);
            if (r) return r;
          }
        } else if (obj && typeof obj === "object") {
          for (const v of Object.values(obj)) {
            const r = dfs(v);
            if (r) return r;
          }
        }
        return null;
      }

      const found = dfs(dailyJson);
      if (!found) return;

      mappingList = found
        .map((m) => ({
          id: Number(m.id),
          name: m.name,
          limit:
            m.limit != null && Number.isFinite(Number(m.limit))
              ? Number(m.limit)
              : null
        }))
        .filter((m) => m.id && m.name);
	      mappingList.sort((a, b) => a.name.localeCompare(b.name));
	      mappingById = new Map(mappingList.map((m) => [m.id, m]));
	      if (peaksLoaded) {
	        renderPeaksTable();
	      }
	    }

	    function buildVolumesFromDaily(dailyJson) {
	      volumes24hById = new Map();
	      if (!dailyJson) return;

	      const candidate =
	        dailyJson &&
	        dailyJson.volumes_24h &&
	        dailyJson.volumes_24h.data &&
	        typeof dailyJson.volumes_24h.data === "object"
	          ? dailyJson.volumes_24h.data
	          : dailyJson && dailyJson.data && typeof dailyJson.data === "object"
	            ? dailyJson.data
	            : dailyJson &&
	                dailyJson.volumes_24h &&
	                typeof dailyJson.volumes_24h === "object"
	              ? dailyJson.volumes_24h
	              : null;

	      if (!candidate || typeof candidate !== "object") return;

	      for (const [idRaw, volRaw] of Object.entries(candidate)) {
	        const id = Number(idRaw);
	        const vol = Number(volRaw);
	        if (!Number.isFinite(id) || !Number.isFinite(vol)) continue;
	        volumes24hById.set(id, vol);
	      }

	      if (peaksLoaded) {
	        renderPeaksTable();
	      }
	    }

	    function renderPinnedList() {
	      const state = loadPinnedState();
	      const entries = Object.entries(state)
        .filter(([, v]) => v && v.pinned)
        .map(([idStr, v]) => ({
          item_id: Number(idStr),
          name: v.name || ("Item " + idStr),
          pinnedAtIso: v.pinnedAtIso || v.starredAtIso || null,
          forecastLen: Array.isArray(v.forecastAtStar)
            ? v.forecastAtStar.length
            : 0
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

    function buildPaginationControls(totalRows, pageSize, currentPage) {
      const totalPages = totalRows > 0 ? Math.ceil(totalRows / pageSize) : 1;
      const clampedPage =
        totalPages > 0 ? Math.min(Math.max(1, currentPage), totalPages) : 1;

      if (clampedPage !== rankingCurrentPage) {
        rankingCurrentPage = clampedPage;
      }

      const container = document.createElement("div");
      container.className = "pagination-controls";

      const startIndex =
        totalRows === 0 ? 0 : (clampedPage - 1) * pageSize + 1;
      const endIndex =
        totalRows === 0
          ? 0
          : Math.min(totalRows, clampedPage * pageSize);

      const left = document.createElement("div");
      left.textContent =
        totalRows === 0
          ? "No items to display."
          : "Showing items " +
            startIndex +
            "–" +
            endIndex +
            " of " +
            totalRows +
            ".";
      container.appendChild(left);

      const right = document.createElement("div");
      right.style.display = "flex";
      right.style.alignItems = "center";
      right.style.gap = "0.5rem";

      const perPageLabel = document.createElement("label");
      perPageLabel.textContent = "Per page:";

      const select = document.createElement("select");
      RANKING_PAGE_SIZE_OPTIONS.forEach((size) => {
        const opt = document.createElement("option");
        opt.value = String(size);
        opt.textContent = String(size);
        if (size === pageSize) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });

      select.addEventListener("change", (ev) => {
        const value = Number(ev.target.value);
        if (Number.isFinite(value) && value > 0) {
          rankingPageSize = value;
          rankingCurrentPage = 1;
          renderTopTable();
        }
      });

      perPageLabel.appendChild(select);
      right.appendChild(perPageLabel);

      const prevBtn = document.createElement("button");
      prevBtn.type = "button";
      prevBtn.textContent = "Previous";
      prevBtn.disabled = clampedPage <= 1;
      prevBtn.addEventListener("click", () => {
        if (rankingCurrentPage > 1) {
          rankingCurrentPage -= 1;
          renderTopTable();
        }
      });
      right.appendChild(prevBtn);

      const nextBtn = document.createElement("button");
      nextBtn.type = "button";
      nextBtn.textContent = "Next";
      nextBtn.disabled = clampedPage >= totalPages || totalRows === 0;
      nextBtn.addEventListener("click", () => {
        if (rankingCurrentPage < totalPages) {
          rankingCurrentPage += 1;
          renderTopTable();
        }
      });
      right.appendChild(nextBtn);

      const pageInfo = document.createElement("div");
      pageInfo.textContent =
        "Page " + clampedPage + " of " + (totalRows === 0 ? 1 : totalPages) + ".";
      right.appendChild(pageInfo);

      container.appendChild(right);

      return container;
    }

    function renderTopTable() {
      if (!overviewSignals.length) {
        tableContainer.textContent = "No signals available.";
        return;
      }
    
      const pins = getPinnedSet();
    
      const rows = overviewSignals
        .map((s) => {
          const id = s.item_id;
          const name = s.name || ("Item " + id);

          const recommendedNotional =
            typeof s.recommended_notional === "number" &&
            Number.isFinite(s.recommended_notional)
              ? s.recommended_notional
              : 0;

          const winProb =
            typeof s.prob_profit === "number" ? s.prob_profit : 0;
          const profit =
            typeof s.expected_profit === "number" ? s.expected_profit : 0;
    
          const gpPerSec =
            typeof s.expected_profit_per_second === "number"
              ? s.expected_profit_per_second
              : null;
          const gpPerHour =
            gpPerSec != null && Number.isFinite(gpPerSec)
              ? gpPerSec * 3600
              : null;
    
          const volWindow =
            typeof s.volume_window === "number" ? s.volume_window : 0;
    
          const liqScore =
            volWindow > 0 ? Math.log10(1 + volWindow) : 0;
    
          // Regime info from signals
          const regime =
            typeof s.regime === "string" && s.regime.length
              ? s.regime
              : null;
          const regimePenalty =
            typeof s.regime_penalty === "number" && Number.isFinite(s.regime_penalty)
              ? s.regime_penalty
              : 1;
    
          // Combined score: Win% × profit × liquidity × regimePenalty
          const score =
            Math.max(0, winProb) *
            Math.max(0, profit) *
            Math.max(1e-3, liqScore || 1) *
            Math.max(0.1, regimePenalty || 1);
    
          return {
            raw: s,
            id,
            name,
            winProb,
            profit,
            gpPerHour,
            volWindow,
            holdMinutes:
              typeof s.hold_minutes === "number" ? s.hold_minutes : null,
            regime,
            regimePenalty,
            recommendedNotional,
            combinedScore: Number.isFinite(score) ? score : 0,
            pinned: pins.has(String(id))
          };
        })
        // Only surface items the sizing policy would actually trade
        .filter(
          (row) =>
            Number.isFinite(row.combinedScore) &&
            row.recommendedNotional > 0
        );
    
      rows.sort((a, b) => b.combinedScore - a.combinedScore);

      const pageSize =
        rankingPageSize && Number.isFinite(rankingPageSize)
          ? rankingPageSize
          : 10;
      const totalRows = rows.length;
      const totalPages =
        totalRows > 0 ? Math.ceil(totalRows / pageSize) : 1;
      if (rankingCurrentPage < 1) {
        rankingCurrentPage = 1;
      } else if (rankingCurrentPage > totalPages) {
        rankingCurrentPage = totalPages;
      }
      const currentPage = rankingCurrentPage;
      const start = totalRows === 0 ? 0 : (currentPage - 1) * pageSize;
      const end = totalRows === 0 ? 0 : Math.min(totalRows, start + pageSize);
      const pageRows = rows.slice(start, end);
    
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");
    
      const trHead = document.createElement("tr");
      [
        "★",
        "Item",
        "Regime",
        "Rec size",
        "Win %",
        "Exp. profit (gp)",
        "GP/hr @limit",
        "Window volume",
        "Hold (m)"
      ].forEach((h) => {
        const th = document.createElement("th");
        th.textContent = h;
        trHead.appendChild(th);
      });
      thead.appendChild(trHead);
    
      pageRows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.className = "clickable";
    
        const tdStar = document.createElement("td");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pin-btn " + (row.pinned ? "pinned" : "unpinned");
        btn.textContent = row.pinned ? "★" : "☆";
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          togglePin(row.id, row.name, row.raw.path || []);
        });
        tdStar.appendChild(btn);
        tr.appendChild(tdStar);
    
        const tdName = document.createElement("td");
        tdName.textContent = row.name;
        tr.appendChild(tdName);
    
        const tdRegime = document.createElement("td");
        if (row.regime === "high") {
          tdRegime.textContent = "H";
          tdRegime.title = "High-value regime (mid_price ≥ 100k)";
        } else if (row.regime === "mid") {
          tdRegime.textContent = "M";
          tdRegime.title = "Mid-value regime (10k ≤ mid_price < 100k)";
        } else if (row.regime === "low") {
          tdRegime.textContent = "L";
          tdRegime.title = "Low-value regime (<10k; noisier / experimental)";
        } else if (row.regime) {
          tdRegime.textContent = row.regime;
          tdRegime.title = "Unknown regime";
        } else {
          tdRegime.textContent = "-";
          tdRegime.title = "No regime info";
        }
        tr.appendChild(tdRegime);

        const tdSize = document.createElement("td");
        tdSize.textContent = row.recommendedNotional.toFixed(2);
        tr.appendChild(tdSize);

        const tdWin = document.createElement("td");
        tdWin.textContent = formatPercent(row.winProb);
        tr.appendChild(tdWin);
    
        const tdProfit = document.createElement("td");
        tdProfit.textContent = formatProfitGp(row.profit);
        tr.appendChild(tdProfit);
    
        const tdHr = document.createElement("td");
        tdHr.textContent =
          row.gpPerHour != null && Number.isFinite(row.gpPerHour)
            ? formatGpPerHour(row.gpPerHour)
            : "-";
        tr.appendChild(tdHr);
    
        const tdVol = document.createElement("td");
        tdVol.textContent =
          row.volWindow != null
            ? row.volWindow.toLocaleString("en-US")
            : "-";
        tr.appendChild(tdVol);
    
        const tdHold = document.createElement("td");
        tdHold.textContent =
          row.holdMinutes != null ? row.holdMinutes + "m" : "-";
        tr.appendChild(tdHold);
    
        tr.addEventListener("click", () => {
          loadPriceSeries(row.id, row.name);
        });
    
        tbody.appendChild(tr);
      });
    
      table.appendChild(thead);
      table.appendChild(tbody);
    
      tableContainer.innerHTML = "";
      const pagerTop = buildPaginationControls(
        totalRows,
        pageSize,
        currentPage
      );
      const pagerBottom = buildPaginationControls(
        totalRows,
        pageSize,
        currentPage
      );
      tableContainer.appendChild(pagerTop);
      tableContainer.appendChild(table);
      tableContainer.appendChild(pagerBottom);
    }

	    function buildPeaksPaginationControls(totalRows, pageSize, currentPage) {
      const totalPages = totalRows > 0 ? Math.ceil(totalRows / pageSize) : 1;
      const clampedPage =
        totalPages > 0 ? Math.min(Math.max(1, currentPage), totalPages) : 1;

      if (clampedPage !== peaksCurrentPage) {
        peaksCurrentPage = clampedPage;
      }

      const container = document.createElement("div");
      container.className = "pagination-controls";

      const startIndex =
        totalRows === 0 ? 0 : (clampedPage - 1) * pageSize + 1;
      const endIndex =
        totalRows === 0
          ? 0
          : Math.min(totalRows, clampedPage * pageSize);

      const left = document.createElement("div");
      left.textContent =
        totalRows === 0
          ? "No items to display."
          : "Showing items " +
            startIndex +
            "–" +
            endIndex +
            " of " +
            totalRows +
            ".";
      container.appendChild(left);

      const right = document.createElement("div");
      right.style.display = "flex";
      right.style.alignItems = "center";
      right.style.gap = "0.5rem";

      const perPageLabel = document.createElement("label");
      perPageLabel.textContent = "Per page:";

      const select = document.createElement("select");
      PEAKS_PAGE_SIZE_OPTIONS.forEach((size) => {
        const opt = document.createElement("option");
        opt.value = String(size);
        opt.textContent = String(size);
        if (size === pageSize) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });

      select.addEventListener("change", (ev) => {
        const value = Number(ev.target.value);
        if (Number.isFinite(value) && value > 0) {
          peaksPageSize = value;
          peaksCurrentPage = 1;
          renderPeaksTable();
        }
      });

      perPageLabel.appendChild(select);
      right.appendChild(perPageLabel);

      const prevBtn = document.createElement("button");
      prevBtn.type = "button";
      prevBtn.textContent = "Previous";
      prevBtn.disabled = clampedPage <= 1;
      prevBtn.addEventListener("click", () => {
        if (peaksCurrentPage > 1) {
          peaksCurrentPage -= 1;
          renderPeaksTable();
        }
      });
      right.appendChild(prevBtn);

      const nextBtn = document.createElement("button");
      nextBtn.type = "button";
      nextBtn.textContent = "Next";
      nextBtn.disabled = clampedPage >= totalPages || totalRows === 0;
      nextBtn.addEventListener("click", () => {
        if (peaksCurrentPage < totalPages) {
          peaksCurrentPage += 1;
          renderPeaksTable();
        }
      });
      right.appendChild(nextBtn);

      const pageInfo = document.createElement("div");
      pageInfo.textContent =
        "Page " + clampedPage + " of " + (totalRows === 0 ? 1 : totalPages) + ".";
      right.appendChild(pageInfo);

      container.appendChild(right);

      return container;
	    }

	    function renderPeaksSortPane(percentColumns) {
	      if (!peaksSortPaneEl || !Array.isArray(percentColumns)) return;

	      const keys = percentColumns.map((c) => c.key);
	      const sameKeys =
	        keys.length === peaksSortPaneKeys.length &&
	        keys.every((k, i) => k === peaksSortPaneKeys[i]);
	      if (sameKeys && peaksSortPaneEl.childElementCount > 0) {
	        return;
	      }
	      peaksSortPaneKeys = keys;

	      peaksSortPaneEl.innerHTML = "";
	      percentColumns.forEach((col, idx) => {
	        const weightExisting = peaksPercentWeights[col.key];
	        const weightVal = Number.isFinite(weightExisting)
	          ? weightExisting
	          : DEFAULT_PEAK_WEIGHT;
	        peaksPercentWeights[col.key] = weightVal;

	        const rowEl = document.createElement("div");
	        rowEl.className = "peaks-weight-row";

	        const labelEl = document.createElement("div");
	        labelEl.className = "peaks-weight-label";
	        const nameSpan = document.createElement("span");
	        nameSpan.textContent = col.header;
	        const valueSpan = document.createElement("span");
	        valueSpan.textContent = String(weightVal);
	        labelEl.appendChild(nameSpan);
	        labelEl.appendChild(valueSpan);

	        const slider = document.createElement("input");
	        slider.type = "range";
	        slider.min = "0";
	        slider.max = "100";
	        slider.step = "1";
	        slider.value = String(weightVal);
	        slider.setAttribute("aria-label", col.header + " weight");
	        slider.id = "peaks-weight-" + idx;
	        slider.addEventListener("input", (ev) => {
	          const v = Number(ev.target.value);
	          peaksPercentWeights[col.key] = Number.isFinite(v) ? v : 0;
	          valueSpan.textContent = String(peaksPercentWeights[col.key]);
	        });

	        rowEl.appendChild(labelEl);
	        rowEl.appendChild(slider);
	        peaksSortPaneEl.appendChild(rowEl);
	      });

	      const applyBtn = document.createElement("button");
	      applyBtn.type = "button";
	      applyBtn.className = "peaks-apply-btn";
	      applyBtn.textContent = "Apply weights";
	      applyBtn.addEventListener("click", () => {
	        peaksSortKey = "__weighted_avg";
	        peaksSortDir = "desc";
	        peaksCurrentPage = 1;
	        renderPeaksTable();
	      });
	      peaksSortPaneEl.appendChild(applyBtn);
	    }

	    function renderPeaksTable() {
	      if (!peaksTableContainer) return;
	      if (!peaksItems.length) {
	        peaksTableContainer.textContent = "No catching‑peaks candidates available.";
	        return;
      }

	      const rows = peaksItems.slice();

	      const pageSize =
	        peaksPageSize && Number.isFinite(peaksPageSize) ? peaksPageSize : 10;

	      const table = document.createElement("table");
	      const thead = document.createElement("thead");
	      const tbody = document.createElement("tbody");

      function getPriceDifference(row) {
        const low = row && row.low_avg_price;
        const peak = row && row.peak_avg_price;
        if (!Number.isFinite(low) || !Number.isFinite(peak)) return null;
        return peak - low;
      }

      function getTradingCap(row) {
        const mapEntry = mappingById.get(Number(row && row.item_id));
        return mapEntry && Number.isFinite(mapEntry.limit) ? mapEntry.limit : null;
      }

      function getVolume24h(row) {
        const itemId = Number(row && row.item_id);
        if (Number.isFinite(itemId)) {
          const dailyVol = volumes24hById.get(itemId);
          if (Number.isFinite(dailyVol)) return dailyVol;
        }
        const v = row && row.volume_24h;
        return Number.isFinite(v) ? v : null;
      }

	      function formatCount(v) {
	        return Number.isFinite(v) ? Math.round(v).toLocaleString("en-US") : "-";
	      }

	      function formatDays(v) {
	        return Number.isFinite(v) ? v.toFixed(1) : "-";
	      }

      const baseColumns = [
        {
          key: "low_avg_price",
          header: "Low Average Price",
          value: (row) => row.low_avg_price,
          format: formatProfitGp
        },
        {
          key: "peak_avg_price",
          header: "Peak Average Price",
          value: (row) => row.peak_avg_price,
          format: formatProfitGp
        },
        {
          key: "price_difference",
          header: "Price Difference",
          value: (row) => getPriceDifference(row),
          format: formatProfitGp
        },
        {
          key: "pct_difference",
          header: "% Difference",
          value: (row) => row.pct_difference,
          format: (v) => (Number.isFinite(v) ? v.toFixed(1) + "%" : "-")
        },
        {
          key: "sharpness",
          header: "Sharpness",
          value: (row) => row.score,
          format: (v) => (Number.isFinite(v) ? v.toFixed(4) : "-")
        },
        {
          key: "peaks_count",
          header: "Peaks Count",
          value: (row) => row.peaks_count,
          format: formatCount
        },
	        {
	          key: "profit_24h",
	          header: "24 Trading Profit",
	          value: (row) => {
	            const diff = getPriceDifference(row);
	            const vol = getVolume24h(row);
	            return Number.isFinite(diff) && Number.isFinite(vol) ? diff * vol : null;
	          },
	          format: formatProfitGp
	        },
	        {
	          key: "volume_24h",
	          header: "Last 24 Trading Volume",
	          value: (row) => getVolume24h(row),
	          format: formatCount
	        },
        {
          key: "profit_cap",
          header: "Cap Trading Profit",
          value: (row) => {
            const diff = getPriceDifference(row);
            const cap = getTradingCap(row);
            return Number.isFinite(diff) && Number.isFinite(cap) ? diff * cap : null;
          },
          format: formatProfitGp
        },
	        {
	          key: "trading_cap",
	          header: "Trading Volume Cap",
	          value: (row) => getTradingCap(row),
	          format: formatCount
	        },
	        {
	          key: "time_since_last_peak_days",
	          header: "Time Since Last Peak",
	          value: (row) => row.time_since_last_peak_days,
	          format: formatDays
	        },
	        {
	          key: "avg_time_between_peaks_days",
	          header: "Average Time Between Peaks",
	          value: (row) => row.avg_time_between_peaks_days,
	          format: formatDays
	        }
	      ];

	      function isBigGap(curr, next) {
	        if (!Number.isFinite(curr) || !Number.isFinite(next)) return false;
	        // Only apply outlier peeling on positive-valued columns. If the
	        // sequence crosses zero/negative, stop peeling to avoid nonsense gaps.
	        if (curr <= 0) return false;
	        if (next < 0) return false;
	        if (next === 0) return curr > 0;
	        return (curr - next) / next > 0.10;
	      }

	      function computeNormalizationStats(valueFn) {
	        const vals = [];
	        rows.forEach((r) => {
	          const v = valueFn(r);
	          if (Number.isFinite(v)) vals.push(v);
	        });
	        if (!vals.length) {
	          return { min: null, maxForNorm: null, range: null, hasExcluded: false };
	        }
	        vals.sort((a, b) => b - a);
	        let excludeCount = 0;
	        while (
	          excludeCount + 1 < vals.length &&
	          isBigGap(vals[excludeCount], vals[excludeCount + 1])
	        ) {
	          excludeCount += 1;
	        }
	        const remaining = vals.slice(excludeCount);
	        const maxForNorm = remaining[0];
	        const min = remaining[remaining.length - 1];
	        const rawRange =
	          Number.isFinite(maxForNorm) && Number.isFinite(min)
	            ? maxForNorm - min
	            : NaN;
	        return {
	          min,
	          maxForNorm,
	          range:
	            Number.isFinite(rawRange) && rawRange !== 0 ? rawRange : null,
	          hasExcluded: excludeCount > 0
	        };
	      }

	      const statsByKey = new Map();
	      baseColumns.forEach((col) => {
	        statsByKey.set(col.key, computeNormalizationStats(col.value));
	      });

	      const invertPercentKeys = new Set([
	        "low_avg_price",
	        "time_since_last_peak_days",
	        "avg_time_between_peaks_days"
	      ]);

		      const percentColumns = baseColumns.map((col) => ({
		        key: col.key + "_norm_pct",
		        header: col.header + " %",
		        value: (row) => {
	          const stats = statsByKey.get(col.key);
	          const v = col.value(row);
	          if (
	            col.key === "avg_time_between_peaks_days" &&
	            row &&
	            Number.isFinite(row.peaks_count) &&
	            Math.round(row.peaks_count) === 1
	          ) {
	            return 100;
	          }
	          if (!Number.isFinite(v) || !stats || !Number.isFinite(stats.maxForNorm))
	            return null;
	          if (stats.hasExcluded && v > stats.maxForNorm + 1e-9) return 100;
	          if (!Number.isFinite(stats.range)) return null;
	          let pct = ((v - stats.min) / stats.range) * 100;
	          if (invertPercentKeys.has(col.key) && Number.isFinite(pct)) {
	            pct = 100 - pct;
	          }
	          if (Number.isFinite(pct)) {
	            pct = Math.max(0, Math.min(100, pct));
	          }
	          return pct;
	        },
		        format: (v) => (Number.isFinite(v) ? v.toFixed(2) + "%" : "-")
		      }));

		      renderPeaksSortPane(percentColumns);

		      const allColumns = baseColumns.concat(percentColumns);

		      function getWeightedAverage(row) {
		        let numerator = 0;
		        let denom = 0;
		        percentColumns.forEach((col) => {
		          const wRaw = peaksPercentWeights[col.key];
		          const w = Number.isFinite(wRaw) ? wRaw : DEFAULT_PEAK_WEIGHT;
		          if (w <= 0) return;
		          denom += w;
		          const v = col.value(row);
		          if (Number.isFinite(v)) {
		            numerator += w * v;
		          }
		        });
		        if (denom <= 0) return null;
		        return numerator / denom;
		      }

		      function getSortValue(row) {
		        if (peaksSortKey === "__weighted_avg") {
		          return getWeightedAverage(row);
		        }
		        if (peaksSortKey === "item") {
		          return (row && row.name) || ("Item " + row.item_id);
		        }
		        const col = allColumns.find((c) => c.key === peaksSortKey);
	        if (!col) return null;
	        return col.value(row);
	      }

	      function compareValues(aVal, bVal) {
	        const aNum = Number.isFinite(aVal) ? aVal : null;
	        const bNum = Number.isFinite(bVal) ? bVal : null;
	        if (aNum != null || bNum != null) {
	          const av = aNum != null ? aNum : -Infinity;
	          const bv = bNum != null ? bNum : -Infinity;
	          return av - bv;
	        }
	        const aStr = aVal == null ? "" : String(aVal).toLowerCase();
	        const bStr = bVal == null ? "" : String(bVal).toLowerCase();
	        if (aStr < bStr) return -1;
	        if (aStr > bStr) return 1;
	        return 0;
	      }

	      rows.sort((a, b) => {
	        const cmp = compareValues(getSortValue(a), getSortValue(b));
	        return peaksSortDir === "asc" ? cmp : -cmp;
	      });

		      const displayColumns = baseColumns;

		      const trHead = document.createElement("tr");
		      const headerDefs = [{ key: "item", header: "Item" }].concat(
		        displayColumns.map((c) => ({ key: c.key, header: c.header }))
		      );
	      headerDefs.forEach((h) => {
	        const th = document.createElement("th");
	        const isActive = peaksSortKey === h.key;
	        const arrow = isActive ? (peaksSortDir === "asc" ? " ▲" : " ▼") : "";
	        th.textContent = h.header + arrow;
	        th.style.cursor = "pointer";
	        th.addEventListener("click", (ev) => {
	          ev.stopPropagation();
	          if (peaksSortKey === h.key) {
	            peaksSortDir = peaksSortDir === "asc" ? "desc" : "asc";
	          } else {
	            peaksSortKey = h.key;
	            peaksSortDir = h.key === "item" ? "asc" : "desc";
	          }
	          peaksCurrentPage = 1;
	          renderPeaksTable();
	        });
	        trHead.appendChild(th);
	      });
	      thead.appendChild(trHead);

	      const totalRows = rows.length;
	      const totalPages = totalRows > 0 ? Math.ceil(totalRows / pageSize) : 1;
	      if (peaksCurrentPage < 1) {
	        peaksCurrentPage = 1;
	      } else if (peaksCurrentPage > totalPages) {
	        peaksCurrentPage = totalPages;
	      }
	      const currentPage = peaksCurrentPage;
	      const start = totalRows === 0 ? 0 : (currentPage - 1) * pageSize;
	      const end = totalRows === 0 ? 0 : Math.min(totalRows, start + pageSize);
	      const pageRows = rows.slice(start, end);

	      pageRows.forEach((row) => {
	        const tr = document.createElement("tr");
	        tr.className = "clickable";

        const tdName = document.createElement("td");
        tdName.textContent = row.name || ("Item " + row.item_id);
        tr.appendChild(tdName);

	        displayColumns.forEach((col) => {
	          const td = document.createElement("td");
	          const val = col.value(row);
	          td.textContent = col.format ? col.format(val) : val;
	          tr.appendChild(td);
        });

        tr.addEventListener("click", () => {
          loadPriceSeries(Number(row.item_id), row.name || ("Item " + row.item_id));
        });

        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);

      peaksTableContainer.innerHTML = "";
      const pagerTop = buildPeaksPaginationControls(
        totalRows,
        pageSize,
        currentPage
      );
      const pagerBottom = buildPeaksPaginationControls(
        totalRows,
        pageSize,
        currentPage
      );
      peaksTableContainer.appendChild(pagerTop);
      peaksTableContainer.appendChild(table);
      peaksTableContainer.appendChild(pagerBottom);
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
      const volMap = new Map();
      const fcMap = new Map();
      const oldFcMap = new Map();

      function getTs(pt) {
        if (!pt) return null;
        const iso = pt.timestamp_iso || pt.timestamp || null;
        if (!iso) return null;
        const ts = Date.parse(iso);
        return Number.isFinite(ts) ? ts : null;
      }

      history.forEach((pt) => {
        const ts = getTs(pt);
        if (ts == null) return;
        tsSet.add(ts);
        histMap.set(ts, pt.price);
        if (pt.volume != null && Number.isFinite(pt.volume)) {
          volMap.set(ts, pt.volume);
        }
      });

      forecast.forEach((pt) => {
        const ts = getTs(pt);
        if (ts == null) return;
        tsSet.add(ts);
        fcMap.set(ts, pt.price);
      });

      const starForecast =
        starInfo && Array.isArray(starInfo.forecastAtStar)
          ? starInfo.forecastAtStar
          : [];
      starForecast.forEach((pt) => {
        const ts = getTs(pt);
        if (ts == null) return;
        tsSet.add(ts);
        oldFcMap.set(ts, pt.price);
      });

      let nowTs = null;
      if (history.length) {
        const ts = getTs(history[history.length - 1]);
        if (ts != null) nowTs = ts;
      } else if (forecast.length) {
        const ts = getTs(forecast[0]);
        if (ts != null) nowTs = ts;
      }

      let starTs = null;
      if (starInfo && starInfo.starredAtIso) {
        const ts = Date.parse(starInfo.starredAtIso);
        if (Number.isFinite(ts)) {
          starTs = ts;
        }
      }

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

      // Use Date objects for a real time axis
      const labels = tsList.map(function (ts) {
        return new Date(ts);
      });

      const histData = [];
      const volumeData = [];
      const fcData = [];
      const oldFcData = [];
      const starMarkerData = [];
      const nowMarkerData = [];

      tsList.forEach(function (ts) {
        histData.push(histMap.has(ts) ? histMap.get(ts) : null);
        volumeData.push(volMap.has(ts) ? volMap.get(ts) : null);

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
        volumeData,
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
        const volumeData = tl.volumeData;
        const fcData = tl.fcData;
        const oldFcData = tl.oldFcData;
        const starMarkerData = tl.starMarkerData;
        const nowMarkerData = tl.nowMarkerData;

        latestVolumeTimeline = Array.isArray(volumeData) ? volumeData.slice() : [];

        const allPrices = []
          .concat(histData, fcData)
          .filter((v) => v != null && Number.isFinite(v));

        let yMin = 0;
        let yMax = 1;
        if (allPrices.length) {
          let rawMin = Math.min.apply(null, allPrices);
          let rawMax = Math.max.apply(null, allPrices);
          const mid = (rawMin + rawMax) / 2;

          const maxSpreadFactor = 50;
          if (mid > 0 && rawMax / mid > maxSpreadFactor) {
            rawMax = mid * maxSpreadFactor;
          }
          if (mid > 0 && mid / rawMin > maxSpreadFactor) {
            rawMin = mid / maxSpreadFactor;
          }

          const pad = (rawMax - rawMin) * 0.1 || mid * 0.1 || 1;
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
            labels: labels,
            datasets: datasets
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
              mode: "index",
              intersect: false,
              axis: "x"
            },
            scales: {
              x: {
                type: "time",
                time: {
                  unit: "hour",
                  stepSize: 1,
                  displayFormats: {
                    hour: "MM-dd HH:mm"
                  }
                },
                ticks: {
                  maxRotation: 0,
                  autoSkip: true
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
              },
              tooltip: {
                callbacks: {
                  label: function (context) {
                    const dsLabel = context.dataset && context.dataset.label ? context.dataset.label : "";
                    const value = context.parsed && typeof context.parsed.y === "number"
                      ? context.parsed.y
                      : null;
                    const idx = context.dataIndex;
                    const vol =
                      Array.isArray(latestVolumeTimeline) &&
                      idx != null &&
                      idx >= 0 &&
                      idx < latestVolumeTimeline.length
                        ? latestVolumeTimeline[idx]
                        : null;

                    let base =
                      dsLabel && value != null
                        ? dsLabel + ": " + value.toLocaleString("en-US")
                        : value != null
                        ? value.toLocaleString("en-US")
                        : dsLabel || "";

                    if (vol != null && Number.isFinite(vol) && vol > 0) {
                      base += " (volume " + Math.round(vol).toLocaleString("en-US") + ")";
                    }

                    return base;
                  }
                }
              },
              zoom: {
                zoom: {
                  wheel: {
                    enabled: true
                  },
                  pinch: {
                    enabled: true
                  },
                  mode: "x"
                },
                pan: {
                  enabled: true,
                  mode: "x",
                  modifierKey: "shift"
                },
                limits: {
                  x: { min: "original", max: "original" }
                }
              }
            }
          }
        });

        chartCanvas.addEventListener("dblclick", function () {
          if (priceChart && typeof priceChart.resetZoom === "function") {
            priceChart.resetZoom();
          }
        });

        const src =
          data.meta && data.meta.source ? data.meta.source : "precomputed";
        const truncated =
          data.meta && typeof data.meta.truncated === "boolean"
            ? data.meta.truncated
            : false;
        const historyLatestIso =
          data.meta && data.meta.history_latest_iso ? data.meta.history_latest_iso : null;
        const latest5mIso =
          data.meta && data.meta.latest_5m_timestamp_iso ? data.meta.latest_5m_timestamp_iso : null;
        const hasForecast = forecast && forecast.length > 1;
        const historySourceText =
          "History source: " + src + (truncated ? " (truncated)" : "") + ".";
        const lastTimestampText = historyLatestIso
          ? " Last price timestamp: " + historyLatestIso + "."
          : latest5mIso
          ? " Last price timestamp: " + latest5mIso + "."
          : "";

        if (!hasForecast) {
          priceStatusEl.textContent =
            "No ML forecast for this item (no entry in the latest /signals snapshot). Showing history only. " +
            historySourceText +
            lastTimestampText;
        } else if (starInfo) {
          priceStatusEl.textContent =
            historySourceText +
            lastTimestampText +
            " Blue = history; green = current forecast; yellow dashed = forecast at pin time.";
        } else {
          priceStatusEl.textContent =
            historySourceText +
            lastTimestampText +
            " Blue = history; green = current forecast (5–120 minute horizons).";
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
        const sigJson = await sigRes.json();
        overviewSignals = Array.isArray(sigJson.signals) ? sigJson.signals : [];
        MODEL_HORIZON =
          typeof sigJson.horizon_minutes === "number"
            ? sigJson.horizon_minutes
            : 60;
        MODEL_TAX =
          typeof sigJson.tax_rate === "number" ? sigJson.tax_rate : 0.02;

	        if (dailyRes.ok) {
	          dailySnapshot = await dailyRes.json();
	          buildMappingFromDaily(dailySnapshot);
	          buildVolumesFromDaily(dailySnapshot);
	        } else {
	          dailySnapshot = null;
	          mappingList = [];
	          volumes24hById = new Map();
	        }

        statusEl.textContent = "";
        metaEl.textContent =
          "Signals computed at " +
          (sigJson.generated_at_iso || "unknown time") +
          " – horizon " +
          MODEL_HORIZON +
          " minutes, tax " +
          (MODEL_TAX * 100).toFixed(1) +
          "%.";

        renderTopTable();
        renderPinnedList();
      } catch (err) {
        console.error("Error loading overview:", err);
        statusEl.textContent = "Error loading overview.";
      }
    }

    async function loadPeaksOverview() {
      if (!peaksStatusEl) return;
      try {
        peaksStatusEl.textContent = "Fetching /catching-peaks...";
        const res = await fetch("/catching-peaks");
        if (!res.ok) {
          peaksStatusEl.textContent =
            "Failed to load catching‑peaks (HTTP " + res.status + ").";
          return;
        }
        const json = await res.json();
        peaksItems = Array.isArray(json.items) ? json.items : [];
        peaksLoaded = true;
        peaksStatusEl.textContent = "";
        if (peaksMetaEl) {
          peaksMetaEl.textContent =
            "Computed at " +
            (json.generated_at_iso || "unknown time") +
            " over ~" +
            (json.window_days || "?") +
            " days.";
        }
        renderPeaksTable();
      } catch (err) {
        console.error("Error loading catching-peaks overview:", err);
        peaksStatusEl.textContent = "Error loading catching‑peaks overview.";
      }
    }

    loadOverview();
    // Restore last active tab and move chart there.
    let savedTab = "standard";
    try {
      const raw = window.localStorage.getItem(ACTIVE_TAB_KEY);
      if (raw === "peaks") savedTab = "peaks";
    } catch (_) {}
    setActiveTab(savedTab);
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

const LATEST_5M_CACHE_TTL_MS = 2 * 60 * 1000; // refresh latest 5m snapshot every ~2 minutes
let LAST_5M_SNAPSHOT = null;
let LAST_5M_FETCHED_AT = 0;
let LAST_5M_KEY = null;

const PEAKS_CACHE_TTL_MS = 15 * 60 * 1000;
let LAST_PEAKS_JSON = null;
let LAST_PEAKS_FETCHED_AT = 0;
// Catching-peaks signals are precomputed by the Python agent and stored in R2
// at signals/peaks/latest.json.

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
        "Attempt " +
          (i + 1) +
          " to fetch " +
          key +
          " failed: " +
          err.message +
          (isLast ? "" : "; retrying in " + delay + "ms")
      );
      if (!isLast) {
        await sleep(delay);
      }
    }
  }
  console.error("Failed to fetch " + key + " after " + attempts + " attempts", lastError);
  return null;
}

async function bucketListWithRetry(env, options, { attempts = 3, baseDelayMs = 200 } = {}) {
  let lastError = null;
  for (let i = 0; i < attempts; i++) {
    try {
      return await env.OSRS_BUCKET.list(options);
    } catch (err) {
      lastError = err;
      const isLast = i === attempts - 1;
      const delay = baseDelayMs * Math.pow(2, i) + Math.random() * 100;
      console.warn(
        "Attempt " +
          (i + 1) +
          " to list " +
          (options && options.prefix ? options.prefix : "") +
          " failed: " +
          err.message +
          (isLast ? "" : "; retrying in " + delay + "ms")
      );
      if (!isLast) {
        await sleep(delay);
      }
    }
  }
  console.error(
    "Failed to list with options " + JSON.stringify(options) + " after " + attempts + " attempts",
    lastError
  );
  return null;
}

async function loadLatestFiveMinuteSnapshot(env) {
  const nowMs = Date.now();
  if (LAST_5M_SNAPSHOT && nowMs - LAST_5M_FETCHED_AT < LATEST_5M_CACHE_TTL_MS) {
    return { snapshot: LAST_5M_SNAPSHOT, key: LAST_5M_KEY };
  }

  const today = new Date();
  for (let delta = 0; delta < 2; delta++) {
    const d = new Date(today.getTime() - delta * 86400000);
    const year = d.getUTCFullYear();
    const month = String(d.getUTCMonth() + 1).padStart(2, "0");
    const day = String(d.getUTCDate()).padStart(2, "0");
    const prefix = "5m/" + year + "/" + month + "/" + day + "/";

    const listing = await bucketListWithRetry(env, { prefix, limit: 1000 });
    if (!listing || !listing.objects || !listing.objects.length) {
      continue;
    }

    const objects = listing.objects.slice().sort((a, b) => (a.key < b.key ? -1 : 1));
    const latest = objects[objects.length - 1];
    const obj = await bucketGetWithRetry(env, latest.key, { attempts: 2, baseDelayMs: 150 });
    if (!obj) continue;

    try {
      const text = await obj.text();
      const parsed = JSON.parse(text);
      LAST_5M_SNAPSHOT = parsed;
      LAST_5M_FETCHED_AT = nowMs;
      LAST_5M_KEY = latest.key;
      return { snapshot: parsed, key: latest.key };
    } catch (err) {
      console.error("Failed to parse latest 5m snapshot for " + latest.key, err);
    }
  }

  return { snapshot: LAST_5M_SNAPSHOT, key: LAST_5M_KEY };
}

function maybeAppendLatestFiveMinute(history, latestSnap, itemId) {
  if (!latestSnap || !latestSnap.five_minute || !latestSnap.five_minute.data) {
    return { history, added: false, latestIso: null };
  }

  const tsSec = Number(latestSnap.five_minute.timestamp);
  const entry = latestSnap.five_minute.data[String(itemId)];
  if (!entry || !Number.isFinite(tsSec)) {
    return { history, added: false, latestIso: null };
  }

  const ah = entry.avgHighPrice;
  const al = entry.avgLowPrice;
  if (typeof ah !== "number" || typeof al !== "number") {
    return { history, added: false, latestIso: null };
  }

  const mid = (ah + al) / 2;
  if (!Number.isFinite(mid) || mid <= 0) {
    return { history, added: false, latestIso: null };
  }

  const highVol = entry.highPriceVolume;
  const lowVol = entry.lowPriceVolume;
  let volume = 0;
  if (typeof highVol === "number" && Number.isFinite(highVol) && highVol > 0) {
    volume += highVol;
  }
  if (typeof lowVol === "number" && Number.isFinite(lowVol) && lowVol > 0) {
    volume += lowVol;
  }

  const tsMs = Math.floor(tsSec * 1000);
  const iso = new Date(tsMs).toISOString();

  let lastTs = null;
  if (history.length) {
    const last = history[history.length - 1];
    const isoStr = last.timestamp_iso || last.timestamp || null;
    const parsed = isoStr ? Date.parse(isoStr) : NaN;
    if (Number.isFinite(parsed)) {
      lastTs = parsed;
    }
  }

  if (lastTs != null && tsMs <= lastTs) {
    return { history, added: false, latestIso: iso };
  }

  const newHistory = history.concat([
    {
      timestamp_iso: iso,
      price: mid,
      ...(volume > 0 ? { volume } : {})
    }
  ]);
  return { history: newHistory, added: true, latestIso: iso };
}

async function loadSignalsWithCache(env) {
  const now = Date.now();
  if (LAST_SIGNALS_JSON && now - LAST_SIGNALS_FETCHED_AT < SIGNALS_CACHE_TTL_MS) {
    return { json: LAST_SIGNALS_JSON, source: "cache" };
  }

  const obj = await bucketGetWithRetry(env, "signals/quantile/latest.json");
  if (obj) {
    try {
      const text = await obj.text();
      const parsed = JSON.parse(text);
      if (parsed && Array.isArray(parsed.signals)) {
        LAST_SIGNALS_JSON = parsed;
        LAST_SIGNALS_FETCHED_AT = Date.now();
        return { json: parsed, source: "fresh" };
      }
    } catch (err) {
      console.error("Error parsing signals/quantile/latest.json:", err);
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
    if (oldestKey !== undefined) {
      PRICE_CACHE.delete(oldestKey);
    }
  }
}

async function handleCatchingPeaks(env) {
  const now = Date.now();
  if (LAST_PEAKS_JSON && now - LAST_PEAKS_FETCHED_AT < PEAKS_CACHE_TTL_MS) {
    return new Response(LAST_PEAKS_JSON, {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  }

  const obj = await bucketGetWithRetry(env, "signals/peaks/latest.json", {
    attempts: 2,
    baseDelayMs: 150
  });
  if (!obj) {
    return new Response(JSON.stringify({ error: "No catching-peaks signals found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" }
    });
  }

  const text = await obj.text();
  LAST_PEAKS_JSON = text;
  LAST_PEAKS_FETCHED_AT = Date.now();

  return new Response(text, {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

async function handleSignals(env) {
  const obj = await bucketGetWithRetry(env, "signals/quantile/latest.json");
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
    return new Response(text, {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  }

  return new Response(JSON.stringify({ error: "No daily snapshot found" }), {
    status: 404,
    headers: { "Content-Type": "application/json" }
  });
}

// Load precomputed per-item history from R2: history/{itemId}.json
async function loadPrecomputedHistory(env, itemId) {
    const key = "history/" + itemId + ".json";
  const obj = await bucketGetWithRetry(env, key, { attempts: 2, baseDelayMs: 150 });
  if (!obj) {
    return { history: [], found: false };
  }

  try {
    const text = await obj.text();
    const parsed = JSON.parse(text);
    const raw = Array.isArray(parsed.history) ? parsed.history : [];

    const cleaned = raw
      .map((pt) => {
        const iso =
          pt.timestamp_iso || pt.timestamp || pt.time_iso || pt.time || null;
        const ts = iso ? Date.parse(iso) : NaN;
        const price =
          typeof pt.price === "number" && Number.isFinite(pt.price)
            ? pt.price
            : NaN;
        const volume =
          typeof pt.volume === "number" && Number.isFinite(pt.volume)
            ? pt.volume
            : null;
        return { ts, iso, price, volume };
      })
      .filter(
        (pt) =>
          Number.isFinite(pt.ts) &&
          pt.iso &&
          typeof pt.iso === "string" &&
          Number.isFinite(pt.price) &&
          pt.price > 0
      )
      .sort((a, b) => a.ts - b.ts)
      .map((pt) => ({
        timestamp_iso: new Date(pt.ts).toISOString(),
        price: pt.price,
        volume: pt.volume
      }));

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
  const s = signals.find(
    (row) => row.item_id === itemId || String(row.item_id) === String(itemId)
  );
  if (!s || !Array.isArray(s.path) || typeof s.mid_now !== "number") {
    return forecast;
  }

  const anchorMid = history.length ? history[history.length - 1].price : s.mid_now;
  if (!Number.isFinite(anchorMid) || anchorMid <= 0) {
    return forecast;
  }

  let baseTimeMs;
  if (history.length) {
    const last = history[history.length - 1];
    const iso = last.timestamp_iso || last.timestamp || null;
    const ts = iso ? Date.parse(iso) : NaN;
    baseTimeMs = Number.isFinite(ts) ? ts : Date.now();
  } else {
    baseTimeMs = Date.now();
  }

  forecast.push({
    timestamp_iso: new Date(baseTimeMs).toISOString(),
    price: anchorMid
  });

  for (const p of s.path) {
    if (
      !p ||
      typeof p.minutes !== "number" ||
      typeof p.future_return_hat !== "number"
    ) {
      continue;
    }

    const minutes = p.minutes;
    let ret = p.future_return_hat;

    ret = Math.max(-0.9, Math.min(3.0, ret));

    const price = anchorMid * (1 + ret);
    if (!Number.isFinite(price) || price <= 0) {
      continue;
    }

    const ts = baseTimeMs + minutes * 60_000;
    if (!Number.isFinite(ts)) continue;

    forecast.push({
      timestamp_iso: new Date(ts).toISOString(),
      price: price
    });
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

  const { history: baseHistory, found } = await loadPrecomputedHistory(env, itemId);
  let history = Array.isArray(baseHistory) ? baseHistory.slice() : [];

  let source = found ? "precomputed" : "missing";
  let latest5mIso = null;
  let latest5mKey = null;

  const latest5m = await loadLatestFiveMinuteSnapshot(env);
  if (latest5m && latest5m.snapshot) {
    latest5mKey = latest5m.key;
    const res = maybeAppendLatestFiveMinute(history, latest5m.snapshot, itemId);
    history = res.history;
    latest5mIso = res.latestIso;
    if (res.added && source === "precomputed") {
      source = "precomputed+latest_5m";
    } else if (res.added && source === "missing") {
      source = "latest_5m_only";
    }
  }

  const historyLatestIso =
    history.length && history[history.length - 1]
      ? history[history.length - 1].timestamp_iso || history[history.length - 1].timestamp || null
      : null;

  const forecast = await buildForecast(env, itemId, history);
  const truncated = !found;

  const body = JSON.stringify({
    item_id: itemId,
    history,
    forecast,
    meta: {
      truncated: truncated,
      source,
      history_latest_iso: historyLatestIso,
      latest_5m_snapshot_key: latest5mKey,
      latest_5m_timestamp_iso: latest5mIso
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
      return new Response(HTML, {
        status: 200,
        headers: { "Content-Type": "text/html; charset=utf-8" }
      });
    }

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

    if (url.pathname === "/catching-peaks") {
      return handleCatchingPeaks(env);
    }

    return new Response("Not found", { status: 404 });
  }
};
