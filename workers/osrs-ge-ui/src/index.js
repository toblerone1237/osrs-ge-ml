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
      max-width: 100%;
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
    .table-scroll {
      overflow-x: auto;
      max-width: 100%;
      -webkit-overflow-scrolling: touch;
    }
    table {
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.35rem 0.45rem;
      border-bottom: 1px solid #1f2937;
      text-align: right;
      white-space: nowrap;
    }
    td.heat {
      background-color: hsla(var(--heat-hue, 120), 65%, 28%, 0.22);
      color: hsl(var(--heat-hue, 120), 75%, 82%);
      transition: background-color 120ms ease, color 120ms ease;
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
    .chart-scrollbar {
      display: none;
      margin-top: 0.35rem;
      align-items: center;
      gap: 0.5rem;
      padding: 0.15rem 0.35rem;
      border-radius: 999px;
      border: 1px solid rgba(55,65,81,0.8);
      background: rgba(15,23,42,0.35);
    }
    .chart-scrollbar input[type="range"] {
      width: 100%;
      margin: 0;
      height: 14px;
      background: transparent;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
      --thumb-width: 48px;
    }
    .chart-scrollbar input[type="range"]:focus {
      outline: none;
    }
    .chart-scrollbar input[type="range"]::-webkit-slider-runnable-track {
      height: 8px;
      border-radius: 999px;
      background: rgba(148,163,184,0.16);
      box-shadow: inset 0 0 0 1px rgba(148,163,184,0.12);
    }
    .chart-scrollbar input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      height: 10px;
      width: var(--thumb-width);
      border-radius: 999px;
      background: rgba(148,163,184,0.55);
      margin-top: -1px;
      box-shadow:
        0 1px 0 rgba(0,0,0,0.35),
        inset 0 0 0 1px rgba(248,250,252,0.18);
    }
    .chart-scrollbar input[type="range"]:hover::-webkit-slider-thumb {
      background: rgba(148,163,184,0.72);
    }
    .chart-scrollbar input[type="range"]:active::-webkit-slider-thumb {
      background: rgba(226,232,240,0.85);
    }
    .chart-scrollbar input[type="range"]:focus-visible::-webkit-slider-thumb {
      box-shadow:
        0 0 0 3px rgba(59,130,246,0.35),
        0 1px 0 rgba(0,0,0,0.35),
        inset 0 0 0 1px rgba(248,250,252,0.18);
    }
    .chart-scrollbar input[type="range"]::-moz-range-track {
      height: 8px;
      border-radius: 999px;
      background: rgba(148,163,184,0.16);
      box-shadow: inset 0 0 0 1px rgba(148,163,184,0.12);
    }
    .chart-scrollbar input[type="range"]::-moz-range-thumb {
      height: 10px;
      width: var(--thumb-width);
      border-radius: 999px;
      border: none;
      background: rgba(148,163,184,0.55);
      box-shadow:
        0 1px 0 rgba(0,0,0,0.35),
        inset 0 0 0 1px rgba(248,250,252,0.18);
    }
    .chart-scrollbar input[type="range"]:hover::-moz-range-thumb {
      background: rgba(148,163,184,0.72);
    }
    .chart-scrollbar input[type="range"]:active::-moz-range-thumb {
      background: rgba(226,232,240,0.85);
    }
    .chart-scrollbar .chart-scroll-label {
      font-size: 0.75rem;
      color: #9ca3af;
      white-space: nowrap;
      user-select: none;
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
		    .peaks-filter-row {
		      display: flex;
		      flex-direction: column;
		      gap: 0.25rem;
		      margin-bottom: 0.7rem;
		    }
		    .peaks-filter-controls {
		      display: flex;
		      gap: 0.35rem;
		      align-items: center;
		    }
		    .peaks-filter-controls select,
		    .peaks-filter-controls input {
		      padding: 0.25rem 0.35rem;
		      border-radius: 0.35rem;
		      border: 1px solid #374151;
		      background: #020617;
		      color: #e5e7eb;
		      font-size: 0.8rem;
		      min-width: 0;
		    }
		    .peaks-filter-controls select {
		      width: 64px;
		      flex: 0 0 auto;
		    }
		    .peaks-filter-controls input {
		      flex: 1;
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
			    .peaks-controls {
			      display: flex;
			      align-items: center;
			      justify-content: space-between;
			      gap: 0.5rem;
			      flex-wrap: wrap;
			      margin-bottom: 0.4rem;
			    }
			    .peaks-search-row {
			      margin-bottom: 0;
			      flex: 1;
			      min-width: 180px;
			    }
			    .peaks-search-row input {
			      min-width: 140px;
			    }
			    .peaks-toggle-btn {
			      padding: 0.3rem 0.6rem;
			      border-radius: 0.35rem;
			      border: 1px solid #4b5563;
		      background: #111827;
		      color: #e5e7eb;
		      cursor: pointer;
		      font-size: 0.8rem;
		    }
		    .peaks-toggle-btn:hover {
		      background: #1f2937;
		    }
		    .peaks-toggle-btn.active {
		      background: #1f2937;
		      border-color: #6b7280;
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
				              Sharpness = average % above the local mean (±3 days) at each peak tip.
				            </div>
			            <div class="peaks-controls">
			              <div class="search-row peaks-search-row">
			                <input id="peaksSearchInput" type="text" placeholder="Filter by item name or id..." />
			                <button id="peaksSearchClearBtn" type="button">Clear</button>
			              </div>
			              <button id="peaksShowAsPctBtn" type="button" class="peaks-toggle-btn">Show As %</button>
			            </div>
			            <div id="peaksSearchStatus" class="small"></div>
			            <div id="peaksTableContainer">Waiting for data...</div>
			          </div>

	          <div id="peaksChartMount"></div>
	        </div>

	        <div class="peaks-tab-side">
	          <div class="card peaks-sort-card">
		            <h3>Sort &amp; filter</h3>
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
      <div id="priceChartScrollWrap" class="chart-scrollbar" aria-hidden="true">
        <span class="chart-scroll-label">Pan</span>
        <input
          id="priceChartScroll"
          type="range"
          min="0"
          max="1000"
          value="0"
          step="1"
          aria-label="Pan chart window"
        />
      </div>
      <div id="priceStatus" class="small" style="margin-top:0.4rem;"></div>
      <div class="small" style="margin-top:0.25rem;">Scroll to zoom, Shift+drag to pan, drag the slider to scroll, double‑click to reset.</div>
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
    const chartScrollWrapEl = document.getElementById("priceChartScrollWrap");
    const chartScrollEl = document.getElementById("priceChartScroll");

    const searchInput = document.getElementById("searchInput");
    const searchButton = document.getElementById("searchButton");
    const searchStatusEl = document.getElementById("searchStatus");
    const searchResultsEl = document.getElementById("searchResults");
    const pinnedListEl = document.getElementById("pinnedList");

			    const peaksStatusEl = document.getElementById("peaksStatus");
			    const peaksMetaEl = document.getElementById("peaksMeta");
			    const peaksTableContainer = document.getElementById("peaksTableContainer");
			    const peaksSortPaneEl = document.getElementById("peaksSortPane");
			    const peaksSearchInput = document.getElementById("peaksSearchInput");
			    const peaksSearchClearBtn = document.getElementById("peaksSearchClearBtn");
			    const peaksSearchStatusEl = document.getElementById("peaksSearchStatus");
			    const peaksShowAsPctBtn = document.getElementById("peaksShowAsPctBtn");
			    const standardChartMount = document.getElementById("standardChartMount");
			    const peaksChartMount = document.getElementById("peaksChartMount");
			    const priceCardEl = document.getElementById("priceCard");
    const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
    const tabStandardEl = document.getElementById("tab-standard");
    const tabPeaksEl = document.getElementById("tab-peaks");

		    const PIN_KEY = "osrs_ge_pins_v3";
		    const ACTIVE_TAB_KEY = "osrs_ge_active_tab_v1";
		    const PEAKS_WEIGHTS_KEY = "osrs_ge_peaks_weights_v1";
		    const PEAKS_FILTERS_KEY = "osrs_ge_peaks_filters_v1";
		    const PEAKS_SHOW_AS_PCT_KEY = "osrs_ge_peaks_show_as_pct_v1";

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
			    let peaksWindowDays = null;
			    let peaksBaselineHalfWindowDays = null;
			    let peaksSortKey = "sharpness";
			    let peaksSortDir = "desc";
			    const DEFAULT_PEAK_WEIGHT = 100;
			    let peaksPercentWeights = {};
			    let peaksColumnFilters = {};
			    let peaksSortPaneMode = "weights";
			    let peaksSortPaneSignature = "";
				    let peaksShowAsPercent = false;
				    let peaksSearchQuery = "";
				    let peaksTableRenderScheduled = false;
			    // Latest volume timeline for the active chart (aligned to labels)
			    let latestVolumeTimeline = [];

			    const CHART_SCROLL_STEPS = 1000;

			    function clampNumber(v, lo, hi) {
			      const n = Number(v);
			      if (!Number.isFinite(n)) return lo;
			      return Math.min(hi, Math.max(lo, n));
			    }

			    function setChartScrollThumbWidth(fullWidth, windowWidth) {
			      if (!chartScrollEl) return;
			      if (!(Number.isFinite(fullWidth) && fullWidth > 0)) return;
			      if (!(Number.isFinite(windowWidth) && windowWidth > 0)) return;

			      const frac = clampNumber(windowWidth / fullWidth, 0, 1);
			      const minPx = 28;
			      const maxPx = 180;
			      const thumbPx = Math.round(
			        clampNumber(minPx + (maxPx - minPx) * frac, minPx, maxPx)
			      );
			      chartScrollEl.style.setProperty("--thumb-width", thumbPx + "px");
			    }

			    function setChartScrollVisible(visible) {
			      if (!chartScrollWrapEl) return;
			      chartScrollWrapEl.style.display = visible ? "flex" : "none";
			      chartScrollWrapEl.setAttribute("aria-hidden", visible ? "false" : "true");
			    }

			    function setChartOriginalXRange(chart) {
			      if (!chart || !chart.scales || !chart.scales.x) return;
			      const x = chart.scales.x;
			      const min = x.min;
			      const max = x.max;
			      if (Number.isFinite(min) && Number.isFinite(max) && max > min) {
			        chart.__xOriginal = { min, max };
			      }
			    }

			    function setChartOriginalYRange(chart) {
			      if (!chart) return;
			      if (chart.__yOriginal) return;
			      const yOpts =
			        chart.options && chart.options.scales && chart.options.scales.y
			          ? chart.options.scales.y
			          : null;
			      if (!yOpts) return;
			      chart.__yOriginal = { min: yOpts.min, max: yOpts.max };
			    }

			    function restoreChartYRange(chart) {
			      if (!chart || !chart.__yOriginal) return;
			      const yOpts =
			        chart.options && chart.options.scales && chart.options.scales.y
			          ? chart.options.scales.y
			          : null;
			      if (!yOpts) return;

			      const nextMin = chart.__yOriginal.min;
			      const nextMax = chart.__yOriginal.max;
			      const curMin = yOpts.min;
			      const curMax = yOpts.max;

			      if (curMin === nextMin && curMax === nextMax) return;
			      yOpts.min = nextMin;
			      yOpts.max = nextMax;
			      try {
			        chart.update("none");
			      } catch (_) {
			        chart.update();
			      }
			    }

			    function labelToUnixMs(label) {
			      if (label == null) return NaN;
			      if (typeof label === "number") return label;
			      if (label instanceof Date) return label.getTime();
			      const t = new Date(label).getTime();
			      return Number.isFinite(t) ? t : NaN;
			    }

			    function updateChartYForVisibleRange(chart, xMin, xMax, isZoomed) {
			      if (!chart) return;
			      const yOpts =
			        chart.options && chart.options.scales && chart.options.scales.y
			          ? chart.options.scales.y
			          : null;
			      if (!yOpts) return;

			      setChartOriginalYRange(chart);

			      if (!isZoomed) {
			        restoreChartYRange(chart);
			        return;
			      }

			      if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMax <= xMin) return;

			      const labels =
			        chart.data && Array.isArray(chart.data.labels) ? chart.data.labels : [];
			      const datasets =
			        chart.data && Array.isArray(chart.data.datasets) ? chart.data.datasets : [];

			      let rawMin = Infinity;
			      let rawMax = -Infinity;

			      for (let i = 0; i < labels.length; i++) {
			        const ts = labelToUnixMs(labels[i]);
			        if (!Number.isFinite(ts)) continue;
			        if (ts < xMin || ts > xMax) continue;

			        for (let d = 0; d < datasets.length; d++) {
			          const ds = datasets[d];
			          if (!ds || ds.hidden) continue;
			          const arr = ds.data;
			          if (!Array.isArray(arr) || i >= arr.length) continue;
			          const v = Number(arr[i]);
			          if (!Number.isFinite(v)) continue;
			          rawMin = Math.min(rawMin, v);
			          rawMax = Math.max(rawMax, v);
			        }
			      }

			      if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) return;

			      const mid = (rawMin + rawMax) / 2;
			      const maxSpreadFactor = 50;
			      if (mid > 0 && rawMax / mid > maxSpreadFactor) {
			        rawMax = mid * maxSpreadFactor;
			      }
			      if (mid > 0 && mid / rawMin > maxSpreadFactor) {
			        rawMin = mid / maxSpreadFactor;
			      }

			      const pad = (rawMax - rawMin) * 0.1 || Math.abs(mid) * 0.05 || 1;
			      const nextMin = Math.max(0, Math.floor(rawMin - pad));
			      const nextMax = Math.ceil(rawMax + pad);
			      if (!(Number.isFinite(nextMin) && Number.isFinite(nextMax) && nextMax > nextMin))
			        return;

			      const curMin = yOpts.min;
			      const curMax = yOpts.max;
			      if (curMin === nextMin && curMax === nextMax) return;

			      yOpts.min = nextMin;
			      yOpts.max = nextMax;
			      try {
			        chart.update("none");
			      } catch (_) {
			        chart.update();
			      }
			    }

			    function updateChartScrollFromChart(chart) {
			      if (!chartScrollEl || !chartScrollWrapEl || !chart) return;
			      const x = chart.scales && chart.scales.x ? chart.scales.x : null;
			      if (!x) return;

			      const original = chart.__xOriginal;
			      const originalMin = original ? original.min : null;
			      const originalMax = original ? original.max : null;
			      const curMin = x.min;
			      const curMax = x.max;

			      if (
			        !Number.isFinite(originalMin) ||
			        !Number.isFinite(originalMax) ||
			        !Number.isFinite(curMin) ||
			        !Number.isFinite(curMax)
			      ) {
			        setChartScrollVisible(false);
			        restoreChartYRange(chart);
			        return;
			      }

			      const fullWidth = originalMax - originalMin;
			      const windowWidth = curMax - curMin;
			      if (!(fullWidth > 0) || !(windowWidth > 0)) {
			        setChartScrollVisible(false);
			        restoreChartYRange(chart);
			        return;
			      }

			      const isZoomed = windowWidth < fullWidth - 1;
			      if (!isZoomed) {
			        setChartScrollVisible(false);
			        updateChartYForVisibleRange(chart, curMin, curMax, false);
			        return;
			      }

			      setChartScrollVisible(true);
			      setChartScrollThumbWidth(fullWidth, windowWidth);
			      updateChartYForVisibleRange(chart, curMin, curMax, true);

			      const maxOffset = fullWidth - windowWidth;
			      const offset = clampNumber(curMin - originalMin, 0, maxOffset);
			      const frac = maxOffset > 0 ? offset / maxOffset : 0;
			      chartScrollEl.value = String(
			        Math.round(clampNumber(frac, 0, 1) * CHART_SCROLL_STEPS)
			      );
			    }

			    function applyChartScrollFromSlider(chart) {
			      if (!chartScrollEl || !chart) return;
			      const x = chart.scales && chart.scales.x ? chart.scales.x : null;
			      if (!x) return;

			      const original = chart.__xOriginal;
			      const originalMin = original ? original.min : null;
			      const originalMax = original ? original.max : null;
			      if (!Number.isFinite(originalMin) || !Number.isFinite(originalMax)) return;

			      const curMin = x.min;
			      const curMax = x.max;
			      if (!Number.isFinite(curMin) || !Number.isFinite(curMax)) return;

			      const fullWidth = originalMax - originalMin;
			      const windowWidth = curMax - curMin;
			      const maxOffset = fullWidth - windowWidth;
			      if (!(maxOffset > 0)) return;

			      const frac = clampNumber(
			        Number(chartScrollEl.value) / CHART_SCROLL_STEPS,
			        0,
			        1
			      );
			      const nextMin = originalMin + frac * maxOffset;
			      const nextMax = nextMin + windowWidth;

			      if (typeof chart.zoomScale === "function") {
			        try {
			          chart.zoomScale("x", { min: nextMin, max: nextMax });
			          updateChartScrollFromChart(chart);
			          return;
			        } catch (_) {}
			      }

			      if (chart.options && chart.options.scales && chart.options.scales.x) {
			        chart.options.scales.x.min = nextMin;
			        chart.options.scales.x.max = nextMax;
			      } else if (x.options) {
			        x.options.min = nextMin;
			        x.options.max = nextMax;
			      }
			      try {
			        chart.update("none");
			      } catch (_) {
			        chart.update();
			      }
			      updateChartScrollFromChart(chart);
			    }

			    if (chartScrollEl) {
			      chartScrollEl.addEventListener("input", function () {
			        if (!priceChart) return;
			        applyChartScrollFromSlider(priceChart);
			      });
			    }

			    if (chartCanvas) {
			      chartCanvas.addEventListener("dblclick", function () {
			        if (priceChart && typeof priceChart.resetZoom === "function") {
			          priceChart.resetZoom();
			          updateChartScrollFromChart(priceChart);
			        }
			      });
			    }

	    function moveChartToTab(tabName) {
	      const mount =
	        tabName === "peaks" ? peaksChartMount : standardChartMount;
	      if (mount && priceCardEl && priceCardEl.parentElement !== mount) {
	        mount.appendChild(priceCardEl);
	      }
	    }

	    function syncForecastVisibilityForTab(tabName) {
	      if (!priceChart || !priceChart.data) return;
	      const hideForecast = tabName === "peaks";
	      const datasets = Array.isArray(priceChart.data.datasets)
	        ? priceChart.data.datasets
	        : [];
	      let changed = false;
	      datasets.forEach((ds) => {
	        if (!ds || typeof ds.label !== "string") return;
	        const isForecast = ds.label.startsWith("Forecast");
	        if (!isForecast) return;
	        if (Boolean(ds.hidden) !== hideForecast) {
	          ds.hidden = hideForecast;
	          changed = true;
	        }
	      });
	      if (changed) {
	        try {
	          priceChart.update("none");
	        } catch (_) {}
	        updateChartScrollFromChart(priceChart);
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
	      syncForecastVisibilityForTab(active);
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

	    function loadPeaksWeights() {
	      try {
	        const raw = window.localStorage.getItem(PEAKS_WEIGHTS_KEY);
	        if (!raw) return {};
	        const obj = JSON.parse(raw);
	        if (!obj || typeof obj !== "object") return {};
	        const out = {};
	        Object.entries(obj).forEach(([k, v]) => {
	          const n = Number(v);
	          if (Number.isFinite(n)) out[k] = n;
	        });
	        return out;
	      } catch (err) {
	        console.warn("Failed to parse peaks weights:", err);
	      }
	      return {};
	    }

		    function savePeaksWeights(weights) {
		      try {
		        window.localStorage.setItem(
		          PEAKS_WEIGHTS_KEY,
		          JSON.stringify(weights || {})
		        );
		      } catch (err) {
		        console.warn("Failed to save peaks weights:", err);
		      }
		    }

		    function loadPeaksFilters() {
		      try {
		        const raw = window.localStorage.getItem(PEAKS_FILTERS_KEY);
		        if (!raw) return {};
		        const obj = JSON.parse(raw);
		        if (!obj || typeof obj !== "object") return {};

		        const allowedOps = new Set([">", ">=", "=", "<=", "<"]);
		        const out = {};
		        Object.entries(obj).forEach(([k, v]) => {
		          if (!v || typeof v !== "object") return;
		          const op = String(v.op || "");
		          const n = Number(v.value);
		          if (!allowedOps.has(op)) return;
		          if (!Number.isFinite(n)) return;
		          out[k] = { op, value: n };
		        });
		        return out;
		      } catch (err) {
		        console.warn("Failed to parse peaks filters:", err);
		      }
		      return {};
		    }

		    function savePeaksFilters(filters) {
		      try {
		        window.localStorage.setItem(
		          PEAKS_FILTERS_KEY,
		          JSON.stringify(filters || {})
		        );
		      } catch (err) {
		        console.warn("Failed to save peaks filters:", err);
		      }
		    }

			    peaksPercentWeights = loadPeaksWeights();
			    peaksColumnFilters = loadPeaksFilters();

			    function loadPeaksShowAsPercent() {
			      try {
			        return window.localStorage.getItem(PEAKS_SHOW_AS_PCT_KEY) === "1";
		      } catch (_) {
		        return false;
		      }
		    }

		    function savePeaksShowAsPercent(v) {
		      try {
		        window.localStorage.setItem(PEAKS_SHOW_AS_PCT_KEY, v ? "1" : "0");
		      } catch (_) {}
		    }

		    function updatePeaksShowAsPercentButton() {
		      if (!peaksShowAsPctBtn) return;
		      peaksShowAsPctBtn.classList.toggle("active", peaksShowAsPercent);
		      peaksShowAsPctBtn.textContent = peaksShowAsPercent ? "Show Raw" : "Show As %";
		    }

		    peaksShowAsPercent = loadPeaksShowAsPercent();
		    updatePeaksShowAsPercentButton();
			    if (peaksShowAsPctBtn) {
			      peaksShowAsPctBtn.addEventListener("click", () => {
			        peaksShowAsPercent = !peaksShowAsPercent;
			        savePeaksShowAsPercent(peaksShowAsPercent);
			        updatePeaksShowAsPercentButton();
			        peaksCurrentPage = 1;
			        renderPeaksTable();
			      });
			    }

			    function normalizePeaksSearchQuery(q) {
			      return (q || "").trim().toLowerCase();
			    }

			    function setPeaksSearchQuery(q) {
			      peaksSearchQuery = normalizePeaksSearchQuery(q);
			      peaksCurrentPage = 1;
			      schedulePeaksTableRender();
			    }

			    if (peaksSearchInput) {
			      peaksSearchInput.addEventListener("input", () => {
			        setPeaksSearchQuery(peaksSearchInput.value);
			      });
			      peaksSearchInput.addEventListener("keydown", (ev) => {
			        if (ev.key === "Enter") {
			          setPeaksSearchQuery(peaksSearchInput.value);
			          renderPeaksTable();
			        }
			      });
			    }
			    if (peaksSearchClearBtn) {
			      peaksSearchClearBtn.addEventListener("click", () => {
			        if (peaksSearchInput) peaksSearchInput.value = "";
			        setPeaksSearchQuery("");
			        try {
			          if (peaksSearchInput) peaksSearchInput.focus();
			        } catch (_) {}
			      });
			    }

			    function schedulePeaksTableRender() {
			      if (peaksTableRenderScheduled) return;
			      peaksTableRenderScheduled = true;
			      const run = () => {
		        peaksTableRenderScheduled = false;
		        renderPeaksTable();
		      };
		      if (typeof requestAnimationFrame === "function") {
		        requestAnimationFrame(run);
		      } else {
		        setTimeout(run, 0);
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

    function toDecileFromPercent(pct) {
      if (!Number.isFinite(pct)) return null;
      const decile = Math.floor(pct / 10);
      return Math.max(0, Math.min(9, decile));
    }

	    function applyDecileHeat(td, pct) {
	      const decile = toDecileFromPercent(pct);
	      if (decile == null) return;
	      const hue = (decile / 9) * 120;
	      td.classList.add("heat");
	      td.style.setProperty("--heat-hue", String(hue));
	    }

	    function computePercentRank(rows, getId, getValue, invert = false) {
	      const pairs = [];
	      rows.forEach((row) => {
	        const id = getId(row);
	        const v = getValue(row);
	        if (!Number.isFinite(id) || !Number.isFinite(v)) return;
	        pairs.push({ id, v });
	      });
	      if (!pairs.length) return new Map();

      pairs.sort((a, b) => a.v - b.v);
      const out = new Map();
      if (pairs.length === 1) {
        out.set(pairs[0].id, 100);
        return out;
      }

      const denom = pairs.length - 1;
      pairs.forEach((p, idx) => {
        let pct = (idx / denom) * 100;
        if (invert) pct = 100 - pct;
        out.set(p.id, pct);
      });
	      return out;
	    }

	    function computePercentRankById(rows, getValue, invert = false) {
	      return computePercentRank(rows, (row) => row && row.id, getValue, invert);
	    }

	    function computePercentRankByItemId(rows, getValue, invert = false) {
	      return computePercentRank(
	        rows,
	        (row) => Number(row && row.item_id),
	        getValue,
	        invert
	      );
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
      const scroll = document.createElement("div");
      scroll.className = "table-scroll";
      scroll.appendChild(table);
      pinnedListEl.innerHTML = "";
      pinnedListEl.appendChild(scroll);
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

	      const pctRecSizeById = computePercentRankById(
	        rows,
	        (r) => r.recommendedNotional
	      );
	      const pctWinById = computePercentRankById(rows, (r) => r.winProb);
	      const pctProfitById = computePercentRankById(rows, (r) => r.profit);
	      const pctGpHrById = computePercentRankById(rows, (r) => r.gpPerHour);
	      const pctVolById = computePercentRankById(rows, (r) => r.volWindow);
	      const pctHoldById = computePercentRankById(
	        rows,
	        (r) => r.holdMinutes,
	        true
	      );

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
	        applyDecileHeat(tdSize, pctRecSizeById.get(row.id));
	        tr.appendChild(tdSize);

	        const tdWin = document.createElement("td");
	        tdWin.textContent = formatPercent(row.winProb);
	        applyDecileHeat(tdWin, pctWinById.get(row.id));
	        tr.appendChild(tdWin);
	    
	        const tdProfit = document.createElement("td");
	        tdProfit.textContent = formatProfitGp(row.profit);
	        applyDecileHeat(tdProfit, pctProfitById.get(row.id));
	        tr.appendChild(tdProfit);
	    
	        const tdHr = document.createElement("td");
	        tdHr.textContent =
	          row.gpPerHour != null && Number.isFinite(row.gpPerHour)
	            ? formatGpPerHour(row.gpPerHour)
	            : "-";
	        applyDecileHeat(tdHr, pctGpHrById.get(row.id));
	        tr.appendChild(tdHr);
	    
	        const tdVol = document.createElement("td");
	        tdVol.textContent =
	          row.volWindow != null
	            ? row.volWindow.toLocaleString("en-US")
	            : "-";
	        applyDecileHeat(tdVol, pctVolById.get(row.id));
	        tr.appendChild(tdVol);
	    
	        const tdHold = document.createElement("td");
	        tdHold.textContent =
	          row.holdMinutes != null ? row.holdMinutes + "m" : "-";
	        applyDecileHeat(tdHold, pctHoldById.get(row.id));
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
      const scroll = document.createElement("div");
      scroll.className = "table-scroll";
      scroll.appendChild(table);
      tableContainer.appendChild(scroll);
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

		    function renderPeaksSortPane({ displayColumns, percentColumns }) {
		      if (!peaksSortPaneEl) return;

		      const mode = peaksSortPaneMode === "filters" ? "filters" : "weights";
		      const columns = mode === "filters" ? displayColumns : percentColumns;
		      if (!Array.isArray(columns)) return;

		      const keys = columns.map((c) => c.key);
		      const signature = mode + ":" + keys.join("|");
		      if (signature === peaksSortPaneSignature && peaksSortPaneEl.childElementCount > 0) {
		        return;
		      }
		      peaksSortPaneSignature = signature;

		      peaksSortPaneEl.innerHTML = "";

		      const allowedOps = [
		        { value: "", label: "off" },
		        { value: ">", label: ">" },
		        { value: ">=", label: ">=" },
		        { value: "=", label: "=" },
		        { value: "<=", label: "<=" },
		        { value: "<", label: "<" }
		      ];

		      if (mode === "filters") {
		        const noteEl = document.createElement("div");
		        noteEl.className = "small";
		        noteEl.style.marginBottom = "0.5rem";

		        const activeCount = Object.values(peaksColumnFilters || {}).filter((v) => {
		          if (!v || typeof v !== "object") return false;
		          if (!["<", "<=", "=", ">=", ">"].includes(v.op)) return false;
		          return Number.isFinite(v.value);
		        }).length;
		        noteEl.textContent =
		          "Filters apply to raw column values." +
		          (activeCount ? " Active: " + activeCount + "." : "");
		        peaksSortPaneEl.appendChild(noteEl);

		        columns.forEach((col, idx) => {
		          const existing = peaksColumnFilters && peaksColumnFilters[col.key];
		          const existingOp =
		            existing && typeof existing.op === "string" ? existing.op : "";
		          const existingValue =
		            existing && Number.isFinite(existing.value) ? existing.value : null;

		          const rowEl = document.createElement("div");
		          rowEl.className = "peaks-filter-row";

		          const labelEl = document.createElement("div");
		          labelEl.className = "peaks-weight-label";
		          const nameSpan = document.createElement("span");
		          nameSpan.textContent = col.header;
		          labelEl.appendChild(nameSpan);
		          rowEl.appendChild(labelEl);

		          const controlsEl = document.createElement("div");
		          controlsEl.className = "peaks-filter-controls";

		          const opSelect = document.createElement("select");
		          opSelect.id = "peaks-filter-op-" + idx;
		          opSelect.setAttribute("aria-label", col.header + " filter operator");
		          allowedOps.forEach((o) => {
		            const opt = document.createElement("option");
		            opt.value = o.value;
		            opt.textContent = o.label;
		            opSelect.appendChild(opt);
		          });
		          opSelect.value = allowedOps.some((o) => o.value === existingOp)
		            ? existingOp
		            : "";

		          const valueInput = document.createElement("input");
		          valueInput.type = "number";
		          valueInput.step = "any";
		          valueInput.placeholder = "value";
		          valueInput.id = "peaks-filter-val-" + idx;
		          valueInput.setAttribute("aria-label", col.header + " filter value");
		          if (existingValue != null) {
		            valueInput.value = String(existingValue);
		          }

		          function updateFilter(saveNow) {
		            const op = String(opSelect.value || "");
		            const raw = String(valueInput.value || "").trim();
		            const v = raw ? Number(raw) : NaN;

		            if (!op || !Number.isFinite(v)) {
		              if (peaksColumnFilters && peaksColumnFilters[col.key]) {
		                delete peaksColumnFilters[col.key];
		              }
		            } else {
		              peaksColumnFilters[col.key] = { op, value: v };
		            }

		            if (saveNow) {
		              savePeaksFilters(peaksColumnFilters);
		            }
		            peaksCurrentPage = 1;
		            schedulePeaksTableRender();
		          }

		          opSelect.addEventListener("change", () => updateFilter(true));
		          valueInput.addEventListener("input", () => updateFilter(false));
		          valueInput.addEventListener("change", () => updateFilter(true));
		          valueInput.addEventListener("keydown", (ev) => {
		            if (ev.key === "Enter") {
		              updateFilter(true);
		              renderPeaksTable();
		            }
		          });

		          controlsEl.appendChild(opSelect);
		          controlsEl.appendChild(valueInput);
		          rowEl.appendChild(controlsEl);
		          peaksSortPaneEl.appendChild(rowEl);
		        });

		        const applyBtn = document.createElement("button");
		        applyBtn.type = "button";
		        applyBtn.className = "peaks-apply-btn";
		        applyBtn.textContent = "Apply filters";
		        applyBtn.addEventListener("click", () => {
		          peaksCurrentPage = 1;
		          renderPeaksTable();
		        });
		        peaksSortPaneEl.appendChild(applyBtn);

		        const clearBtn = document.createElement("button");
		        clearBtn.type = "button";
		        clearBtn.className = "peaks-apply-btn";
		        clearBtn.textContent = "Clear filters";
		        clearBtn.addEventListener("click", () => {
		          peaksColumnFilters = {};
		          savePeaksFilters(peaksColumnFilters);
		          peaksCurrentPage = 1;
		          peaksSortPaneSignature = "";
		          renderPeaksTable();
		        });
		        peaksSortPaneEl.appendChild(clearBtn);

		        const backBtn = document.createElement("button");
		        backBtn.type = "button";
		        backBtn.className = "peaks-apply-btn";
		        backBtn.textContent = "Back to weights";
		        backBtn.addEventListener("click", () => {
		          peaksSortPaneMode = "weights";
		          peaksSortPaneSignature = "";
		          renderPeaksTable();
		        });
		        peaksSortPaneEl.appendChild(backBtn);

		        return;
		      }

		      columns.forEach((col, idx) => {
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
		          peaksSortKey = "__weighted_avg";
		          peaksSortDir = "desc";
		          peaksCurrentPage = 1;
		          schedulePeaksTableRender();
		        });
		        slider.addEventListener("change", () => {
		          savePeaksWeights(peaksPercentWeights);
		        });

		        rowEl.appendChild(labelEl);
		        rowEl.appendChild(slider);
		        peaksSortPaneEl.appendChild(rowEl);
		      });
		      savePeaksWeights(peaksPercentWeights);

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

		      const filterBtn = document.createElement("button");
		      filterBtn.type = "button";
		      filterBtn.className = "peaks-apply-btn";
		      filterBtn.textContent = "Filter mode";
		      filterBtn.addEventListener("click", () => {
		        peaksSortPaneMode = "filters";
		        peaksSortPaneSignature = "";
		        renderPeaksTable();
		      });
		      peaksSortPaneEl.appendChild(filterBtn);
		    }

		    function renderPeaksTable() {
		      if (!peaksTableContainer) return;
		      if (!peaksItems.length) {
		        if (peaksSearchStatusEl) peaksSearchStatusEl.textContent = "";
		        peaksTableContainer.textContent = "No catching‑peaks candidates available.";
		        return;
	      }

			      const allRows = peaksItems.slice();
			      const q =
			        typeof peaksSearchQuery === "string"
			          ? peaksSearchQuery.trim().toLowerCase()
			          : "";
			      let rows = allRows;
			      if (q) {
			        rows = allRows.filter((row) => {
			          const idStr =
			            row && row.item_id != null ? String(row.item_id) : "";
			          const name =
			            row && row.name ? String(row.name).toLowerCase() : "";
			          return (
			            (idStr && idStr.includes(q)) ||
			            (name && name.includes(q))
			          );
			        });
			      }
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
	          format: (v) => (Number.isFinite(v) ? v.toFixed(1) + "%" : "-")
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
		          value: (row) => {
		            if (!row) return null;
		            const peaksCount = row.peaks_count;
		            if (
		              Number.isFinite(peaksCount) &&
		              Math.round(peaksCount) <= 1
		            ) {
		              return null;
		            }
		            return row.avg_time_between_peaks_days;
		          },
			          format: formatDays
			        }
			      ];

		      const invertPercentKeys = new Set([
		        "low_avg_price",
		        "time_since_last_peak_days",
		        "avg_time_between_peaks_days"
		      ]);

			      const pctByItemIdByKey = new Map();
			      baseColumns.forEach((col) => {
			        pctByItemIdByKey.set(
			          col.key,
			          computePercentRankByItemId(
			            allRows,
			            col.value,
			            invertPercentKeys.has(col.key)
			          )
			        );
			      });

					      const percentColumns = baseColumns.map((col) => ({
					        key: col.key + "_norm_pct",
					        header: col.header + " %",
					        value: (row) => {
					          const itemId = Number(row && row.item_id);
					          if (!Number.isFinite(itemId)) return null;
					          const byId = pctByItemIdByKey.get(col.key);
					          if (!byId) return null;
					          const pct = byId.get(itemId);
					          return Number.isFinite(pct) ? pct : null;
						        },
						        format: (v) => (Number.isFinite(v) ? v.toFixed(2) + "%" : "-")
						      }));

			      const pctValueFnByBaseKey = new Map();
			      percentColumns.forEach((col) => {
			        const suffix = "_norm_pct";
			        if (typeof col.key === "string" && col.key.endsWith(suffix)) {
			          const baseKey = col.key.slice(0, -suffix.length);
			          pctValueFnByBaseKey.set(baseKey, col.value);
			        }
			      });

				      renderPeaksSortPane({ displayColumns: baseColumns, percentColumns });

			      const allColumns = baseColumns.concat(percentColumns);

			      const activeFilters = Object.entries(peaksColumnFilters || {}).filter(
			        ([, f]) =>
			          f &&
			          typeof f === "object" &&
			          ["<", "<=", "=", ">=", ">"].includes(f.op) &&
			          Number.isFinite(f.value)
			      );

			      if (activeFilters.length) {
			        const colByKey = new Map();
			        baseColumns.forEach((col) => {
			          colByKey.set(col.key, col);
			        });

			        const EPS = 1e-9;
			        rows = rows.filter((row) => {
			          for (const [key, f] of activeFilters) {
			            const col = colByKey.get(key);
			            if (!col) continue;

			            const target = Number(f.value);
			            const rawVal = col.value(row);
			            if (!Number.isFinite(rawVal) || !Number.isFinite(target)) return false;
			            const v = rawVal;

			            switch (f.op) {
			              case ">":
			                if (!(v > target)) return false;
			                break;
			              case ">=":
			                if (!(v >= target)) return false;
			                break;
			              case "<":
			                if (!(v < target)) return false;
			                break;
			              case "<=":
			                if (!(v <= target)) return false;
			                break;
			              case "=":
			                if (!(Math.abs(v - target) <= EPS)) return false;
			                break;
			              default:
			                break;
			            }
			          }
			          return true;
			        });
			      }

			      if (peaksSearchStatusEl) {
			        const hasSearch = Boolean(q);
			        const hasFilters = activeFilters.length > 0;
			        if (!hasSearch && !hasFilters) {
			          peaksSearchStatusEl.textContent = allRows.length + " items.";
			        } else {
			          const parts = [];
			          if (hasSearch) parts.push("search");
			          if (hasFilters) {
			            parts.push(
			              activeFilters.length +
			                " column filter" +
			                (activeFilters.length === 1 ? "" : "s")
			            );
			          }
			          peaksSearchStatusEl.textContent =
			            "Filtered: " +
			            rows.length +
			            " / " +
			            allRows.length +
			            " items." +
			            (parts.length ? " (" + parts.join(", ") + ")" : "");
			        }
			      }

			      if (!rows.length) {
			        peaksTableContainer.textContent = "No items match that filter.";
			        return;
			      }

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
		        th.textContent = h.header;
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
		          const pctValueFn = pctValueFnByBaseKey.get(col.key);
		          const pctValue = pctValueFn ? pctValueFn(row) : null;
		          if (peaksShowAsPercent) {
		            td.textContent = Number.isFinite(pctValue)
		              ? pctValue.toFixed(2) + "%"
		              : "-";
		            applyDecileHeat(td, pctValue);
		          } else {
		            const val = col.value(row);
		            td.textContent = col.format ? col.format(val) : val;
		            applyDecileHeat(td, pctValue);
		          }
		          tr.appendChild(td);
	        });

		        tr.addEventListener("click", () => {
		          loadPriceSeries(Number(row.item_id), row.name || ("Item " + row.item_id), {
		            showForecast: false,
		            highlightPeaks: true,
		            peakBaselinePrice: row.low_avg_price,
		            peakBaselineHalfWindowDays: peaksBaselineHalfWindowDays,
		            peakExpectedCount: row.peaks_count,
		            peakWindowDays: peaksWindowDays
		          });
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
      const scroll = document.createElement("div");
      scroll.className = "table-scroll";
      scroll.appendChild(table);
      peaksTableContainer.appendChild(scroll);
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

		    function countPeakWindows(mask) {
		      if (!Array.isArray(mask) || !mask.length) return 0;
		      let count = 0;
		      let prev = false;
		      for (let i = 0; i < mask.length; i++) {
		        const cur = Boolean(mask[i]);
		        if (cur && !prev) count += 1;
		        prev = cur;
		      }
		      return count;
		    }

		    function filterPeakMaskBySurroundingAverage(histData, mask, factor) {
		      if (
		        !Array.isArray(histData) ||
		        !Array.isArray(mask) ||
		        histData.length !== mask.length
		      ) {
		        return mask;
		      }

		      const mult = Number.isFinite(factor) && factor > 0 ? factor : 2;
		      const out = mask.map(Boolean);
		      const n = out.length;

		      let i = 0;
		      while (i < n) {
		        if (!out[i]) {
		          i += 1;
		          continue;
		        }
		        const start = i;
		        while (i < n && out[i]) i += 1;
		        const end = i - 1;
		        const width = end - start + 1;
		        if (width <= 0) continue;

		        let peakSum = 0;
		        let peakCount = 0;
		        for (let j = start; j <= end; j++) {
		          const v = histData[j];
		          if (v != null && Number.isFinite(v) && v > 0) {
		            peakSum += v;
		            peakCount += 1;
		          }
		        }

		        let surroundSum = 0;
		        let surroundCount = 0;
		        for (let j = start - width; j <= start - 1; j++) {
		          if (j < 0 || j >= n) continue;
		          const v = histData[j];
		          if (v != null && Number.isFinite(v) && v > 0) {
		            surroundSum += v;
		            surroundCount += 1;
		          }
		        }
		        for (let j = end + 1; j <= end + width; j++) {
		          if (j < 0 || j >= n) continue;
		          const v = histData[j];
		          if (v != null && Number.isFinite(v) && v > 0) {
		            surroundSum += v;
		            surroundCount += 1;
		          }
		        }

		        let keep = false;
		        if (peakCount > 0 && surroundCount > 0) {
		          const peakAvg = peakSum / peakCount;
		          const surroundAvg = surroundSum / surroundCount;
		          keep =
		            Number.isFinite(peakAvg) &&
		            Number.isFinite(surroundAvg) &&
		            surroundAvg > 0 &&
		            peakAvg > mult * surroundAvg;
		        }

		        if (!keep) {
		          for (let j = start; j <= end; j++) out[j] = false;
		        }
		      }

		      return out;
		    }

		    function computePeakMaskFixedBaseline(histData, baselinePrice, opts) {
		      if (!Array.isArray(histData) || !histData.length) return null;
		      const startMult =
		        opts && Number.isFinite(opts.startMult) ? opts.startMult : 1.5;
		      const endMult =
		        opts && Number.isFinite(opts.endMult) ? opts.endMult : 1.1;
		      const wantDiagnostics = Boolean(opts && opts.returnDiagnostics);

		      let base = Number(baselinePrice);
		      if (!Number.isFinite(base) || base <= 0) {
		        let sum = 0;
		        let count = 0;
		        histData.forEach((v) => {
		          if (!Number.isFinite(v)) return;
		          sum += v;
		          count += 1;
		        });
		        if (count > 0 && Number.isFinite(sum)) {
		          base = sum / count;
		        }
		      }
		      if (!Number.isFinite(base) || base <= 0) return null;

		      const minPeakPrice = base * startMult;
		      const peakEndPrice = base * endMult;

		      const mask = new Array(histData.length).fill(false);
		      const baselineByIndex = wantDiagnostics
		        ? new Array(histData.length).fill(null)
		        : null;
		      const ratioByIndex = wantDiagnostics
		        ? new Array(histData.length).fill(null)
		        : null;
		      let inPeak = false;
		      for (let i = 0; i < histData.length; i++) {
		        const price = histData[i];
		        if (!Number.isFinite(price)) {
		          if (inPeak) inPeak = false;
		          continue;
		        }
		        if (baselineByIndex) baselineByIndex[i] = base;
		        if (ratioByIndex) ratioByIndex[i] = price / base;
		        if (!inPeak) {
		          if (price >= minPeakPrice) {
		            inPeak = true;
		            mask[i] = true;
		          }
		          continue;
		        }
		        if (price <= peakEndPrice) {
		          inPeak = false;
		          continue;
		        }
		        mask[i] = true;
		      }
		      const filteredMask = filterPeakMaskBySurroundingAverage(histData, mask, 2);
		      if (wantDiagnostics) {
		        return { mask: filteredMask, baselineByIndex, ratioByIndex };
		      }
		      return filteredMask;
		    }

		    function computePeakMaskLocalMean(labels, histData, opts) {
		      if (
		        !Array.isArray(labels) ||
		        !Array.isArray(histData) ||
		        labels.length !== histData.length
		      ) {
		        return null;
		      }

		      const halfWindowDays =
		        opts && Number.isFinite(opts.halfWindowDays) ? opts.halfWindowDays : null;
		      if (!(halfWindowDays > 0)) return null;

		      const startMult =
		        opts && Number.isFinite(opts.startMult) ? opts.startMult : 1.5;
		      const endMult =
		        opts && Number.isFinite(opts.endMult) ? opts.endMult : 1.1;
		      const wantDiagnostics = Boolean(opts && opts.returnDiagnostics);

		      const halfWindowMs = halfWindowDays * 86400 * 1000;

		      let idx = [];
		      let ts = [];
		      let prices = [];
		      for (let i = 0; i < histData.length; i++) {
		        const p = histData[i];
		        if (!Number.isFinite(p) || p <= 0) continue;
		        const d = labels[i];
		        const t =
		          d instanceof Date
		            ? d.getTime()
		            : typeof d === "string"
		              ? Date.parse(d)
		              : NaN;
		        if (!Number.isFinite(t)) continue;
		        idx.push(i);
		        ts.push(t);
		        prices.push(p);
		      }

		      const windowDays =
		        opts && Number.isFinite(opts.windowDays) ? opts.windowDays : null;
		      if (windowDays != null && windowDays > 0 && ts.length) {
		        const cutoff = ts[ts.length - 1] - windowDays * 86400 * 1000;
		        if (Number.isFinite(cutoff)) {
		          const fIdx = [];
		          const fTs = [];
		          const fPrices = [];
		          for (let i = 0; i < prices.length; i++) {
		            if (ts[i] >= cutoff) {
		              fIdx.push(idx[i]);
		              fTs.push(ts[i]);
		              fPrices.push(prices[i]);
		            }
		          }
		          idx = fIdx;
		          ts = fTs;
		          prices = fPrices;
		        }
		      }
		      if (prices.length < 2) return null;

		      // Sliding mean within ±halfWindowMs (matches ml/score_catching_peaks.py)
		      const localMean = new Array(prices.length).fill(NaN);
		      let left = 0;
		      let right = 0;
		      let windowSum = 0;

		      for (let i = 0; i < prices.length; i++) {
		        const center = ts[i];
		        const rightBound = center + halfWindowMs;
		        const leftBound = center - halfWindowMs;

		        while (right < prices.length && ts[right] <= rightBound) {
		          windowSum += prices[right];
		          right += 1;
		        }
		        while (left < prices.length && ts[left] < leftBound) {
		          windowSum -= prices[left];
		          left += 1;
		        }

		        const count = right - left;
		        if (count > 0) {
		          localMean[i] = windowSum / count;
		        }
		      }

		      const mask = new Array(histData.length).fill(false);
		      const baselineByIndex = wantDiagnostics
		        ? new Array(histData.length).fill(null)
		        : null;
		      const ratioByIndex = wantDiagnostics
		        ? new Array(histData.length).fill(null)
		        : null;
		      let inPeak = false;
		      for (let i = 0; i < prices.length; i++) {
		        const mean = localMean[i];
		        if (!Number.isFinite(mean) || mean <= 0) {
		          if (inPeak) inPeak = false;
		          continue;
		        }
		        const ratio = prices[i] / mean;
		        if (!Number.isFinite(ratio)) {
		          if (inPeak) inPeak = false;
		          continue;
		        }
		        const outIndex = idx[i];
		        if (baselineByIndex && outIndex != null) baselineByIndex[outIndex] = mean;
		        if (ratioByIndex && outIndex != null) ratioByIndex[outIndex] = ratio;

		        if (!inPeak) {
		          if (ratio >= startMult) {
		            inPeak = true;
		            mask[outIndex] = true;
		          }
		          continue;
		        }

		        if (ratio <= endMult) {
		          inPeak = false;
		          continue;
		        }
		        mask[outIndex] = true;
		      }

		      const filteredMask = filterPeakMaskBySurroundingAverage(histData, mask, 2);
		      if (wantDiagnostics) {
		        return { mask: filteredMask, baselineByIndex, ratioByIndex };
		      }
		      return filteredMask;
		    }

		    function computePeakMask(labels, histData, opts) {
		      const localMask = computePeakMaskLocalMean(labels, histData, opts);
		      if (localMask) return localMask;
		      const baselinePrice = opts ? Number(opts.baselinePrice) : NaN;
		      return computePeakMaskFixedBaseline(histData, baselinePrice, opts);
		    }

		    const avgLineOverlayPlugin = {
		      id: "avgLineOverlay",
		      beforeDatasetsDraw(chart, args, pluginOptions) {
		        const value = pluginOptions ? pluginOptions.value : null;
		        if (!Number.isFinite(value)) return;
		        const chartArea = chart && chart.chartArea ? chart.chartArea : null;
		        const yScale = chart && chart.scales ? chart.scales.y : null;
		        if (!chartArea || !yScale) return;
		        const y = yScale.getPixelForValue(value);
		        if (!Number.isFinite(y)) return;
		        if (y < chartArea.top || y > chartArea.bottom) return;

		        const color =
		          pluginOptions && typeof pluginOptions.color === "string"
		            ? pluginOptions.color
		            : "rgba(234,179,8,0.35)";
		        const lineWidth =
		          pluginOptions && Number.isFinite(pluginOptions.lineWidth)
		            ? pluginOptions.lineWidth
		            : 1.5;
		        const dash =
		          pluginOptions && Array.isArray(pluginOptions.dash)
		            ? pluginOptions.dash
		            : [6, 4];

		        const ctx = chart.ctx;
		        ctx.save();
		        ctx.strokeStyle = color;
		        ctx.lineWidth = lineWidth;
		        ctx.setLineDash(dash);
		        ctx.beginPath();
		        ctx.moveTo(chartArea.left, y);
		        ctx.lineTo(chartArea.right, y);
		        ctx.stroke();
		        ctx.restore();
		      },
		      afterDatasetsDraw(chart, args, pluginOptions) {
		        const value = pluginOptions ? pluginOptions.value : null;
		        if (!Number.isFinite(value)) return;
		        const chartArea = chart && chart.chartArea ? chart.chartArea : null;
		        const yScale = chart && chart.scales ? chart.scales.y : null;
		        if (!chartArea || !yScale) return;
		        let y = yScale.getPixelForValue(value);
		        if (!Number.isFinite(y)) return;
		        const ctx = chart.ctx;

		        const text =
		          pluginOptions && typeof pluginOptions.text === "string"
		            ? pluginOptions.text
		            : "";
		        if (!text) return;

		        const padding =
		          pluginOptions && Number.isFinite(pluginOptions.padding)
		            ? pluginOptions.padding
		            : 3;
		        const xPadding = padding;
		        const yPadding = padding;
		        const x = chartArea.right - 4;

		        const fontSize =
		          pluginOptions && Number.isFinite(pluginOptions.fontSize)
		            ? pluginOptions.fontSize
		            : 11;
		        const font =
		          pluginOptions && typeof pluginOptions.font === "string"
		            ? pluginOptions.font
		            : "600 " + fontSize + "px system-ui, -apple-system, Segoe UI, Roboto, Arial";
		        const textColor =
		          pluginOptions && typeof pluginOptions.textColor === "string"
		            ? pluginOptions.textColor
		            : "rgba(234,179,8,0.9)";
		        const bgColor =
		          pluginOptions && typeof pluginOptions.backgroundColor === "string"
		            ? pluginOptions.backgroundColor
		            : "rgba(2,6,23,0.75)";

		        const yMin = chartArea.top + fontSize / 2 + yPadding;
		        const yMax = chartArea.bottom - fontSize / 2 - yPadding;
		        y = Math.max(yMin, Math.min(yMax, y));

		        ctx.save();
		        ctx.font = font;
		        ctx.textAlign = "right";
		        ctx.textBaseline = "middle";
		        const metrics = ctx.measureText(text);
		        const textWidth = metrics && Number.isFinite(metrics.width) ? metrics.width : 0;
		        const boxW = textWidth + xPadding * 2;
		        const boxH = fontSize + yPadding * 2;
		        const boxLeft = x - boxW;
		        const boxTop = y - boxH / 2;
		        ctx.fillStyle = bgColor;
		        ctx.fillRect(boxLeft, boxTop, boxW, boxH);
		        ctx.fillStyle = textColor;
		        ctx.fillText(text, x - xPadding, y);
		        ctx.restore();
		      }
		    };

		    function computeBucketVolumeWindow(labels, volumeData, histData, windowMs) {
		      if (!Array.isArray(labels) || !labels.length) return null;
		      if (!Array.isArray(volumeData) || !volumeData.length) return null;

	      function getLabelTs(label) {
	        if (label instanceof Date) return label.getTime();
	        const parsed = Date.parse(String(label));
	        return Number.isFinite(parsed) ? parsed : null;
	      }

	      let endTs = null;
	      if (Array.isArray(histData)) {
	        for (let i = histData.length - 1; i >= 0; i--) {
	          if (Number.isFinite(histData[i])) {
	            endTs = getLabelTs(labels[i]);
	            break;
	          }
	        }
	      }
	      if (endTs == null) {
	        for (let i = volumeData.length - 1; i >= 0; i--) {
	          if (Number.isFinite(volumeData[i])) {
	            endTs = getLabelTs(labels[i]);
	            break;
	          }
	        }
	      }
	      if (endTs == null) {
	        endTs = getLabelTs(labels[labels.length - 1]);
	      }
	      if (endTs == null) return null;

	      const startTs = endTs - windowMs;
	      let sum = 0;
	      let bucketsInWindow = 0;
	      let bucketsWithVolume = 0;
	      const n = Math.min(labels.length, volumeData.length);
	      for (let i = 0; i < n; i++) {
	        const ts = getLabelTs(labels[i]);
	        if (ts == null) continue;
	        if (ts < startTs || ts > endTs) continue;
	        bucketsInWindow += 1;
	        const v = volumeData[i];
	        if (Number.isFinite(v)) {
	          sum += v;
	          bucketsWithVolume += 1;
	        }
	      }

	      return { startTs, endTs, sum, bucketsInWindow, bucketsWithVolume };
	    }

		    async function loadPriceSeries(itemId, name, opts) {
		      priceTitleEl.textContent = "Price for " + name + " (id " + itemId + ")";
		      priceStatusEl.textContent = "Loading price series...";

	      try {
	        const showForecast = !(opts && opts.showForecast === false);
	        const res = await fetch("/price-series?item_id=" + encodeURIComponent(itemId));
	        if (!res.ok) {
	          priceStatusEl.textContent =
	            "No price data available (HTTP " + res.status + ").";
          if (priceChart) {
            priceChart.destroy();
            priceChart = null;
          }
	          setChartScrollVisible(false);
          return;
        }
	
	        const data = await res.json();
	        const history = data.history || [];
	        const forecast = showForecast ? data.forecast || [] : [];

	        if (!history.length && !forecast.length) {
	          priceStatusEl.textContent = "No price data yet for this item.";
	          if (priceChart) {
            priceChart.destroy();
            priceChart = null;
          }
	          setChartScrollVisible(false);
          return;
        }

	        const pinnedState = loadPinnedState();
	        const pinEntry = pinnedState[String(itemId)];
	        const starInfo =
	          showForecast && pinEntry && pinEntry.pinned
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
	        const volWindow24h = computeBucketVolumeWindow(
	          labels,
	          volumeData,
	          histData,
	          24 * 60 * 60 * 1000
	        );
	        const dailyVol24h = volumes24hById.get(Number(itemId));
	        let volumeInfoText = " Tooltip shows per-bucket volume (5m last 24h, 30m older).";
	        if (
	          volWindow24h &&
	          Number.isFinite(volWindow24h.sum) &&
	          volWindow24h.bucketsWithVolume > 0
	        ) {
	          volumeInfoText +=
	            " Sum last 24h ≈ " +
	            Math.round(volWindow24h.sum).toLocaleString("en-US") +
	            " (" +
	            volWindow24h.bucketsWithVolume +
	            " buckets).";
	        }
		        if (Number.isFinite(dailyVol24h) && dailyVol24h > 0) {
		          volumeInfoText +=
		            " Daily 24h = " +
		            Math.round(dailyVol24h).toLocaleString("en-US") +
		            ".";
		        }

		        let avgPrice = null;
		        {
		          let sum = 0;
		          let count = 0;
		          histData.forEach((v) => {
		            if (!Number.isFinite(v)) return;
		            sum += v;
		            count += 1;
		          });
		          if (count > 0 && Number.isFinite(sum)) {
		            avgPrice = sum / count;
		          }
		        }

		        const allPrices = []
		          .concat(histData)
		          .concat(showForecast ? oldFcData : [])
		          .concat(showForecast ? fcData : [])
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

	        const highlightPeaks = Boolean(opts && opts.highlightPeaks);
	        const peakBaselinePrice = opts ? Number(opts.peakBaselinePrice) : NaN;
	        const peakBaselineHalfWindowDays = opts
	          ? Number(opts.peakBaselineHalfWindowDays)
	          : NaN;
	        const peakExpectedCount = opts ? Number(opts.peakExpectedCount) : NaN;
	        const peakWindowDays = opts ? Number(opts.peakWindowDays) : NaN;

	        let peakRefPrice = peakBaselinePrice;
	        if (!(Number.isFinite(peakRefPrice) && peakRefPrice > 0)) {
	          peakRefPrice =
	            avgPrice != null && Number.isFinite(avgPrice) && avgPrice > 0
	              ? avgPrice
	              : NaN;
	        }

	        const peakDiag = highlightPeaks
	          ? (function () {
	              const baseOpts = {
	                baselinePrice: peakRefPrice,
	                halfWindowDays: peakBaselineHalfWindowDays,
	                windowDays: peakWindowDays,
	                startMult: 1.5,
	                endMult: 1.1,
	                returnDiagnostics: true
	              };

	              const local = computePeakMaskLocalMean(labels, histData, baseOpts);
	              if (local && local.mask && Array.isArray(local.mask)) return local;

	              const fixed = computePeakMaskFixedBaseline(histData, peakRefPrice, baseOpts);
	              if (fixed && fixed.mask && Array.isArray(fixed.mask)) return fixed;

	              return null;
	            })()
	          : null;
	        const peakMask =
	          highlightPeaks && peakDiag && Array.isArray(peakDiag.mask)
	            ? peakDiag.mask
	            : null;
	        const peakBaselineByIndex =
	          highlightPeaks && peakDiag && Array.isArray(peakDiag.baselineByIndex)
	            ? peakDiag.baselineByIndex
	            : null;
	        const peakRatioByIndex =
	          highlightPeaks && peakDiag && Array.isArray(peakDiag.ratioByIndex)
	            ? peakDiag.ratioByIndex
	            : null;
	        const peakWindowCount =
	          highlightPeaks && peakMask ? countPeakWindows(peakMask) : 0;
	        const hasPeakSegments = peakWindowCount > 0;

	        const historyLineColor = "rgba(59,130,246,1)";
	        const peakLineColor = "rgba(16,185,129,1)";

	        const historyDataset = {
	          label: "Historical mid price (5m last 24h, 30m older)",
	          data: histData,
	          borderColor: historyLineColor,
	          backgroundColor: "rgba(59,130,246,0.2)",
	          pointRadius: 0,
	          borderWidth: 2,
	          tension: 0.15,
	          spanGaps: true
	        };

	        if (hasPeakSegments) {
	          historyDataset.segment = {
	            borderColor: (ctx) => {
	              const i0 = ctx && Number.isFinite(ctx.p0DataIndex) ? ctx.p0DataIndex : null;
	              const i1 = ctx && Number.isFinite(ctx.p1DataIndex) ? ctx.p1DataIndex : null;
	              const p0IsPeak =
	                i0 != null && i0 >= 0 && i0 < peakMask.length && peakMask[i0];
	              const p1IsPeak =
	                i1 != null && i1 >= 0 && i1 < peakMask.length && peakMask[i1];
	              return p0IsPeak || p1IsPeak ? peakLineColor : historyLineColor;
	            }
	          };
	        }

	        const datasets = [historyDataset];

	        if (showForecast && starInfo && oldFcData.some((v) => v != null)) {
	          datasets.push({
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

	        const hasCurrentForecastData =
	          showForecast && fcData.some((v) => v != null && Number.isFinite(v));
	        if (hasCurrentForecastData) {
	          datasets.push({
	            label: "Forecast price (next 2h, 5m steps)",
	            data: fcData,
	            borderColor: "rgba(16,185,129,1)",
	            backgroundColor: "rgba(16,185,129,0.15)",
	            pointRadius: 0,
	            borderWidth: 2,
	            borderDash: [6, 3],
	            tension: 0.15,
	            spanGaps: true
	          });
	        }

	        if (showForecast && starInfo && starMarkerData.some((v) => v != null)) {
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
	          plugins: [avgLineOverlayPlugin],
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
	              avgLineOverlay: {
	                value: avgPrice,
	                text:
	                  avgPrice != null && Number.isFinite(avgPrice)
	                    ? "avg " + formatGpPerHour(avgPrice)
	                    : "",
	                color: "rgba(234,179,8,0.35)",
	                dash: [6, 4],
	                lineWidth: 1.5,
	                padding: 3,
	                fontSize: 11,
	                textColor: "rgba(234,179,8,0.9)",
	                backgroundColor: "rgba(2,6,23,0.75)"
	              },
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
	                      base +=
	                        " (bucket vol " + Math.round(vol).toLocaleString("en-US") + ")";
	                    }

	                    if (
	                      highlightPeaks &&
	                      peakBaselineByIndex &&
	                      peakRatioByIndex &&
	                      typeof dsLabel === "string" &&
	                      dsLabel.startsWith("Historical mid price")
	                    ) {
	                      const baseline =
	                        idx != null &&
	                        idx >= 0 &&
	                        idx < peakBaselineByIndex.length
	                          ? peakBaselineByIndex[idx]
	                          : null;
	                      const ratio =
	                        idx != null && idx >= 0 && idx < peakRatioByIndex.length
	                          ? peakRatioByIndex[idx]
	                          : null;
	                      if (
	                        baseline != null &&
	                        Number.isFinite(baseline) &&
	                        baseline > 0 &&
	                        ratio != null &&
	                        Number.isFinite(ratio)
	                      ) {
	                        base +=
	                          " (baseline " +
	                          Math.round(baseline).toLocaleString("en-US") +
	                          ", x" +
	                          ratio.toFixed(2) +
	                          ")";
	                      }
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
                  mode: "x",
                  onZoomComplete: (ctx) =>
                    updateChartScrollFromChart(ctx && ctx.chart ? ctx.chart : null)
                },
                pan: {
                  enabled: true,
                  mode: "x",
                  modifierKey: "shift",
                  onPanComplete: (ctx) =>
                    updateChartScrollFromChart(ctx && ctx.chart ? ctx.chart : null)
                },
                limits: {
                  x: { min: "original", max: "original" }
                }
              }
	            }
	          }
	        });

	        setChartScrollVisible(false);
	        setChartOriginalXRange(priceChart);
	        updateChartScrollFromChart(priceChart);
	        const chartRef = priceChart;
	        setTimeout(() => {
	          if (priceChart !== chartRef) return;
	          setChartOriginalXRange(chartRef);
	          updateChartScrollFromChart(chartRef);
	        }, 0);

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
	        const hasForecast = showForecast && forecast && forecast.length > 1;
	        const historySourceText =
	          "History source: " + src + (truncated ? " (truncated)" : "") + ".";
        const lastTimestampText = historyLatestIso
          ? " Last price timestamp: " + historyLatestIso + "."
          : latest5mIso
          ? " Last price timestamp: " + latest5mIso + "."
          : "";

			        if (!showForecast) {
			          let peakInfoText = "";
			          if (highlightPeaks) {
			            peakInfoText =
			              " Peak windows (green): shown " +
			              peakWindowCount +
			              (Number.isFinite(peakExpectedCount)
			                ? ", table " + Math.round(peakExpectedCount)
			                : "") +
			              (Number.isFinite(peakWindowDays) && peakWindowDays > 0
			                ? ", window " + peakWindowDays + "d"
			                : "") +
			              (Number.isFinite(peakBaselineHalfWindowDays) &&
			              peakBaselineHalfWindowDays > 0
			                ? ", baseline ±" + peakBaselineHalfWindowDays + "d"
			                : "") +
			              "; starts at +50%, ends at +10%).";
			          }
			          priceStatusEl.textContent =
			            historySourceText +
			            lastTimestampText +
			            volumeInfoText +
			            " Forecast hidden on Catching Peaks view." +
			            peakInfoText;
			        } else if (!hasForecast) {
			          priceStatusEl.textContent =
			            "No ML forecast for this item (no entry in the latest /signals snapshot). Showing history only. " +
			            historySourceText +
		            lastTimestampText +
	            volumeInfoText;
	        } else if (starInfo) {
	          priceStatusEl.textContent =
	            historySourceText +
	            lastTimestampText +
	            volumeInfoText +
	            " Blue = history; green = current forecast; yellow dashed = forecast at pin time.";
	        } else {
	          priceStatusEl.textContent =
	            historySourceText +
	            lastTimestampText +
	            volumeInfoText +
	            " Blue = history; green = current forecast (5–120 minute horizons).";
	        }
      } catch (err) {
        console.error("Error loading price series:", err);
        priceStatusEl.textContent = "Error loading price series.";
        if (priceChart) {
          priceChart.destroy();
          priceChart = null;
        }
	        setChartScrollVisible(false);
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
	        peaksWindowDays = Number.isFinite(json.window_days) ? json.window_days : null;
	        peaksBaselineHalfWindowDays = Number.isFinite(json.baseline_half_window_days)
	          ? json.baseline_half_window_days
	          : null;
        peaksStatusEl.textContent = "";
        if (peaksMetaEl) {
          const baselineHalf = Number.isFinite(json.baseline_half_window_days)
            ? json.baseline_half_window_days
            : null;
          peaksMetaEl.textContent =
            "Computed at " +
            (json.generated_at_iso || "unknown time") +
            " over ~" +
            (json.window_days || "?") +
            " days." +
            (baselineHalf != null ? " Peak baseline: ±" + baselineHalf + " days." : "");
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

  function asPositiveNumber(v) {
    const n = typeof v === "number" ? v : Number(v);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  function asNonNegativeNumber(v) {
    const n = typeof v === "number" ? v : Number(v);
    return Number.isFinite(n) && n >= 0 ? n : null;
  }

  const highVol = asNonNegativeNumber(entry.highPriceVolume) || 0;
  const lowVol = asNonNegativeNumber(entry.lowPriceVolume) || 0;
  const volume = highVol + lowVol;

  const ah = asPositiveNumber(entry.avgHighPrice);
  const al = asPositiveNumber(entry.avgLowPrice);

  let mid = null;
  if (ah != null && al != null) {
    if (highVol <= 0 && lowVol > 0) {
      mid = al;
    } else if (lowVol <= 0 && highVol > 0) {
      mid = ah;
    } else {
      mid = (ah + al) / 2;
    }
  } else if (ah != null) {
    mid = ah;
  } else if (al != null) {
    mid = al;
  }

  if (!Number.isFinite(mid) || mid <= 0) {
    return { history, added: false, latestIso: null };
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
      volume: volume
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
