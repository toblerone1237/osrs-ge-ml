// OSRS Grand Exchange daily snapshot worker
// Worker name: osrs-ge-daily

// OSRS Wiki API base
const OSRS_API_BASE = "https://prices.runescape.wiki/api/v1/osrs";

// IMPORTANT: put a real contact in here; they ask for a real User-Agent.
const USER_AGENT = "osrs-ge-collector/1.0 (contact: toblerone1237@gmail.com)";

// Helper: format YYYY/MM/DD
function formatDateParts(date) {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return { year, month, day };
}

// Helper: fetch JSON from OSRS API with headers
async function fetchOsrsJson(path) {
  const url = `${OSRS_API_BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "User-Agent": USER_AGENT,
      "Accept": "application/json",
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`OSRS API error for ${path}: HTTP ${res.status} ${res.statusText} ${text}`);
  }

  return res.json();
}

// Main daily job
async function runDailySnapshot(env) {
  const now = new Date();
  const { year, month, day } = formatDateParts(now);
  const fetchedAtUnix = Math.floor(now.getTime() / 1000);

  // Fetch all three daily endpoints in parallel:
  //  - /24h for last 24h average prices/volumes
  //  - /volumes for total 24h volume per item
  //  - /mapping for item metadata (names, limits, etc.)
  const [daily24h, volumes24h, mapping] = await Promise.all([
    fetchOsrsJson("/24h"),
    fetchOsrsJson("/volumes"),
    fetchOsrsJson("/mapping"),
  ]);

  const snapshot = {
    type: "osrs_ge_snapshot_daily",
    fetched_at_iso: now.toISOString(),
    fetched_at_unix: fetchedAtUnix,
    daily_24h: daily24h,     // directly from API
    volumes_24h: volumes24h, // directly from API
    mapping: mapping,        // array of {id, name, limit, ...}
  };

  const body = JSON.stringify(snapshot);

  // Store under date-based key, e.g. daily/2025/11/15.json
  const keyDated = `daily/${year}/${month}/${day}.json`;
  const keyLatest = "daily/latest.json";

  await env.OSRS_BUCKET.put(keyDated, body, {
    httpMetadata: { contentType: "application/json" },
  });

  await env.OSRS_BUCKET.put(keyLatest, body, {
    httpMetadata: { contentType: "application/json" },
  });

  console.log(`Stored daily snapshot at ${keyDated} and updated daily/latest.json`);
}

// Export handlers
export default {
  // For debugging via URL
  async fetch(request, env, ctx) {
    return new Response("osrs-ge-daily worker. Use cron trigger.", {
      status: 200,
      headers: { "Content-Type": "text/plain" },
    });
  },

  // Called by cron trigger
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runDailySnapshot(env));
  },
};
