// OSRS Grand Exchange 5-minute snapshot worker
// Worker name: osrs-ge-5m
// R2 binding: env.OSRS_BUCKET -> bucket "osrs-ge-raw"

// === CONFIG ===

// Base URL for the OSRS Wiki price API
const OSRS_API_BASE = "https://prices.runescape.wiki/api/v1/osrs";

// IMPORTANT: change this to something that identifies you (they ask for a real User-Agent).
// Example: "osrs-ge-collector/1.0 (contact: your_email_or_discord)"
const USER_AGENT = "osrs-ge-collector/1.0 (contact: toblerone1237@gmail.com)";

// Helper: format a Date as e.g. 2025/11/14/12-35
function formatDatePath(date) {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  const hour = String(date.getUTCHours()).padStart(2, "0");
  const minute = String(date.getUTCMinutes()).padStart(2, "0");
  return { year, month, day, hour, minute };
}

// Helper: pause for the given milliseconds
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Helper: fetch JSON from OSRS API with proper headers and simple retries
async function fetchOsrsJson(path, { attempts = 3, baseDelayMs = 500 } = {}) {
  const url = `${OSRS_API_BASE}${path}`;
  let lastError = null;

  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(url, {
        headers: {
          "User-Agent": USER_AGENT,
          "Accept": "application/json",
        },
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(
          `OSRS API error for ${path}: HTTP ${res.status} ${res.statusText} – ${text}`,
        );
      }

      return await res.json();
    } catch (err) {
      lastError = err;
      const isLast = i === attempts - 1;
      const delay = baseDelayMs * Math.pow(2, i) + Math.random() * 200;
      console.warn(
        `Attempt ${i + 1} for ${path} failed: ${err.message}${isLast ? "" : `; retrying in ${delay}ms`}`,
      );
      if (!isLast) {
        await sleep(delay);
      }
    }
  }

  throw lastError || new Error(`Failed to fetch ${path}`);
}

// Main job: run once per cron to store a 5-minute snapshot
async function runFiveMinuteSnapshot(env) {
  // Current time in UTC (when the Worker runs)
  const now = new Date();

  // Fetch aggregated 5-minute data for all items
  const fiveMinute = await fetchOsrsJson("/5m");

  // Fetch latest observed high/low for all items
  const latest = await fetchOsrsJson("/latest");

  // Build a combined object so each snapshot has everything in one file
  const snapshot = {
    type: "osrs_ge_snapshot_5m",
    fetched_at_iso: now.toISOString(),
    fetched_at_unix: Math.floor(now.getTime() / 1000),
    five_minute: fiveMinute,
    latest: latest,
  };

  // Build an R2 key that sorts nicely and is easy to parse later
  // Example: 5m/2025/11/14/12-35.json
  const { year, month, day, hour, minute } = formatDatePath(now);
  const key = `5m/${year}/${month}/${day}/${hour}-${minute}.json`;

  // Store JSON in R2
  await env.OSRS_BUCKET.put(key, JSON.stringify(snapshot), {
    httpMetadata: {
      contentType: "application/json",
    },
  });

  // Optional: simple logging (visible in Worker Logs)
  console.log(`Stored 5m snapshot at key: ${key}`);
}

// Export the Worker entrypoints
export default {
  // HTTP handler – mainly for manual tests
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === "/run-once") {
      // Manually trigger one snapshot via HTTP
      await runFiveMinuteSnapshot(env);
      return new Response("Manually ran 5m snapshot.\n", { status: 200 });
    }

    return new Response(
      "osrs-ge-5m worker. This endpoint is mainly for diagnostics.\n",
      { status: 200 },
    );
  },

  // Cron handler – called by Cron Triggers (configured later)
  async scheduled(controller, env, ctx) {
    console.log("Cron trigger fired for osrs-ge-5m:", controller.scheduledTime);
    await runFiveMinuteSnapshot(env);
  },
};
