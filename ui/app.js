// App shell: hash router + shared topbar health/strategies polling.
import { api, activeBannerHtml } from "./lib.js";
import { render as renderSettings } from "./pages/settings.js";
import { render as renderIngestion } from "./pages/ingestion.js";
import { render as renderRetrieval } from "./pages/retrieval.js";
import { render as renderQuery } from "./pages/query.js";
import { render as renderIndices } from "./pages/indices.js";
import { render as renderEval } from "./pages/eval.js";

const ROUTES = {
  settings: renderSettings,
  ingestion: renderIngestion,
  retrieval: renderRetrieval,
  query: renderQuery,
  indices: renderIndices,
  eval: renderEval,
};

function currentRoute() {
  const hash = location.hash.replace(/^#\/?/, "");
  return ROUTES[hash] ? hash : "ingestion";
}

async function navigate() {
  const route = currentRoute();
  document.querySelectorAll("#nav a").forEach(a => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  const view = document.getElementById("view");
  view.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  refreshActiveBanner();  // fire-and-forget: the top wiring read-out, independent of page render
  try {
    await ROUTES[route](view);
  } catch (e) {
    view.innerHTML = `<div class="card">Failed to render page: ${e.message}</div>`;
    console.error(e);
  }
}

// Populate the always-on active-config banner from /api/v1/config's resolved `active` block.
async function refreshActiveBanner() {
  const host = document.getElementById("active-banner");
  if (!host) return;
  try {
    const cfg = await api("/api/v1/config");
    host.innerHTML = activeBannerHtml(cfg.active);
  } catch {
    host.innerHTML = "";  // config unreachable — hide the banner rather than show a broken one
  }
}

async function refreshHealth() {
  const dot = document.getElementById("health-dot");
  const label = document.getElementById("health-label");
  try {
    await api("/health");
    dot.className = "dot ok";
    label.textContent = "service healthy";
  } catch {
    dot.className = "dot bad";
    label.textContent = "service unreachable";
  }
  try {
    const s = await api("/api/v1/strategies");
    document.getElementById("strategies-label").textContent = `strategies: ${s.strategies.join(", ")}`;
  } catch {
    document.getElementById("strategies-label").textContent = "strategies: unknown";
  }
}

window.addEventListener("hashchange", navigate);
window.addEventListener("DOMContentLoaded", () => {
  if (!location.hash) location.hash = "#/ingestion";
  navigate();
  refreshHealth();
  setInterval(refreshHealth, 15000);
});
