// Indices → Visualizer sub-tab: a 2D PCA scatter of an index's chunk embeddings, colored by
// bucket or by source document (toggle), with a hover tooltip showing the chunk's truncated
// text and basic stats. Read-only — no writes. Lazily built the first time the Visualizer tab
// is opened (see indices.js).
import { api, errorMessage, escapeHtml } from "../lib.js";

// Categorical palette (dataviz skill's validated dark-mode steps — this UI is dark-only, no
// prefers-color-scheme toggle in styles.css). Fixed order, never cycled: groups beyond the
// 8th fold into a shared "Other" slot rather than repeating a hue (color must follow entity).
const PALETTE = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];
const NO_VALUE_COLOR = "#898781"; // dataviz skill's "muted (axis/labels)" dark-mode ink
const OTHER_COLOR = "#5c6479"; // this UI's --text-faint — distinct from both the palette and no-value gray

// Point field + labels for each color-by mode.
const COLOR_MODES = {
  bucket: { field: "bucket", label: "Bucket", noValueLabel: "(no bucket)" },
  document: { field: "file_name", label: "Document", noValueLabel: "(no document)" },
};

// Map each group name to a color: first 8 (sorted) groups get their own palette slot, any
// beyond that share OTHER_COLOR so a hue is never reused for two different identities.
function buildColorMap(keys) {
  const map = new Map();
  keys.forEach((k, i) => map.set(k, i < PALETTE.length ? PALETTE[i] : OTHER_COLOR));
  return map;
}

function colorForKey(key, colorMap) {
  if (!key) return NO_VALUE_COLOR;
  return colorMap.get(key) || OTHER_COLOR;
}

// Distinct, sorted non-null values of `field` across all points (used for document mode,
// where the server doesn't precompute a distinct list the way it does for buckets).
function distinctValues(points, field) {
  return [...new Set(points.map(p => p[field]).filter(Boolean))].sort();
}

// Fit point (x, y) extents into an SVG viewBox with a margin, so the scatter always fills
// the available space regardless of the PCA projection's raw scale.
function fitToViewBox(points, width, height, margin) {
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scaleX = (width - 2 * margin) / spanX;
  const scaleY = (height - 2 * margin) / spanY;
  return points.map(p => ({
    ...p,
    cx: margin + (p.x - minX) * scaleX,
    cy: margin + (p.y - minY) * scaleY,
  }));
}

function legendHtml(keys, colorMap, noValueLabel, groupNoun) {
  const shown = keys.slice(0, PALETTE.length);
  const overflow = keys.length > PALETTE.length;
  const items = shown.map(k => `<span class="chip-row-item"><span class="legend-swatch" style="background:${colorMap.get(k)}"></span>${escapeHtml(k)}</span>`);
  if (overflow) items.push(`<span class="chip-row-item"><span class="legend-swatch" style="background:${OTHER_COLOR}"></span>Other (${keys.length - shown.length} more ${groupNoun})</span>`);
  items.push(`<span class="chip-row-item"><span class="legend-swatch" style="background:${NO_VALUE_COLOR}"></span>${escapeHtml(noValueLabel)}</span>`);
  return `<div class="chip-row" style="margin-top:.6rem;gap:.9rem">${items.join("")}</div>`;
}

// Recessive grid: hairline lines only, no ticks/labels — a spatial reference, not data itself.
function gridLinesHtml(width, height, margin, cols = 6, rows = 4) {
  const gridColor = "#2c2c2a"; // dataviz skill's dark-mode gridline hairline
  const lines = [];
  for (let i = 0; i <= cols; i++) {
    const x = margin + (i * (width - 2 * margin)) / cols;
    lines.push(`<line x1="${x.toFixed(2)}" y1="${margin}" x2="${x.toFixed(2)}" y2="${height - margin}" stroke="${gridColor}" stroke-width="1"></line>`);
  }
  for (let i = 0; i <= rows; i++) {
    const y = margin + (i * (height - 2 * margin)) / rows;
    lines.push(`<line x1="${margin}" y1="${y.toFixed(2)}" x2="${width - margin}" y2="${y.toFixed(2)}" stroke="${gridColor}" stroke-width="1"></line>`);
  }
  return lines.join("");
}

// Renders the scatter for the given color-by mode ("bucket" | "document"). `onModeChange` is
// called with the newly picked mode when a toggle button is clicked, so the caller can re-render
// without refetching (the already-fetched `data` carries both bucket and file_name per point).
function renderScatter(host, data, colorBy, onModeChange) {
  const width = 720, height = 480, margin = 24;
  if (!data.points.length) {
    host.innerHTML = `<div class="empty-state">Not enough chunks to visualize yet (need at least 2).</div>`;
    return;
  }
  const mode = COLOR_MODES[colorBy];
  const keys = colorBy === "bucket" ? data.buckets : distinctValues(data.points, "file_name");
  const colorMap = buildColorMap(keys);
  const placed = fitToViewBox(data.points, width, height, margin);
  const grid = gridLinesHtml(width, height, margin);
  const circles = placed.map(p => `
    <circle cx="${p.cx.toFixed(2)}" cy="${p.cy.toFixed(2)}" r="4"
      fill="${colorForKey(p[mode.field], colorMap)}" fill-opacity="0.8" stroke="none"
      data-bucket="${escapeHtml(p.bucket || "")}" data-file="${escapeHtml(p.file_name || "")}"
      data-text="${escapeHtml(p.text)}" data-length="${p.length}"></circle>`).join("");
  const capNote = data.capped ? `<p class="hint">showing ${data.returned} of ${data.total} chunks</p>` : `<p class="hint">${data.returned} chunk(s)</p>`;
  const toggleHtml = Object.entries(COLOR_MODES).map(([key, m]) =>
    `<button type="button" class="mode-btn${key === colorBy ? " active" : ""}" data-color-mode="${key}" style="padding:.35rem .8rem;font-size:.85rem">${m.label}</button>`).join("");
  host.innerHTML = `
    <div class="btn-row" style="align-items:center;gap:.5rem;margin-bottom:.6rem">
      <span class="hint">Color by</span>
      <div class="mode-toggle" id="viz-color-toggle" style="display:inline-flex">${toggleHtml}</div>
    </div>
    ${capNote}
    <div class="visualizer-scatter-wrap" style="position:relative">
      <svg viewBox="0 0 ${width} ${height}" width="100%" style="max-width:${width}px;border:1px solid var(--bg-raised);border-radius:6px;background:var(--bg-raised)">
        ${grid}
        ${circles}
      </svg>
      <div class="visualizer-tooltip" style="display:none;position:absolute;pointer-events:none;background:var(--bg,#0d0f14);color:var(--text,#e6e9f0);border:1px solid var(--bg-raised);padding:.5rem .7rem;border-radius:6px;max-width:280px;font-size:.85rem;z-index:5"></div>
    </div>
    ${legendHtml(keys, colorMap, mode.noValueLabel, colorBy === "bucket" ? "buckets" : "documents")}`;

  host.querySelectorAll("#viz-color-toggle .mode-btn").forEach(btn => {
    btn.onclick = () => onModeChange(btn.dataset.colorMode);
  });

  // Hover wiring: one shared tooltip element, repositioned/filled per circle on hover.
  const tooltip = host.querySelector(".visualizer-tooltip");
  const wrap = host.querySelector(".visualizer-scatter-wrap");
  const positionTooltip = (e) => {
    const rect = wrap.getBoundingClientRect();
    tooltip.style.left = `${e.clientX - rect.left + 12}px`;
    tooltip.style.top = `${e.clientY - rect.top + 12}px`;
  };
  host.querySelectorAll("circle").forEach(circle => {
    circle.addEventListener("mouseenter", (e) => {
      const { bucket, file, text, length } = circle.dataset;
      tooltip.innerHTML = `
        <div><strong>${bucket ? escapeHtml(bucket) : "(no bucket)"}</strong></div>
        ${file ? `<div>${escapeHtml(file)}</div>` : ""}
        <div class="hint">${length} chars</div>
        <div style="margin-top:.3rem">${escapeHtml(text)}${Number(length) > text.length ? "…" : ""}</div>`;
      tooltip.style.display = "";
      positionTooltip(e);
    });
    circle.addEventListener("mousemove", positionTooltip);
    circle.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
  });
}

// Build the Visualizer sub-tab: index picker + scatter/legend, fetched fresh on each pick.
export async function buildVisualizerTab(view) {
  const host = document.getElementById("idx-visualizer");

  let indices = [];
  try { indices = (await api("/api/v1/indices")).indices || []; } catch { /* none yet */ }

  host.innerHTML = `
    <div class="card">
      <h2 style="margin-top:0">📊 Visualizer</h2>
      <p class="page-sub" style="margin-top:0">2D projection (PCA) of an index's chunk embeddings, colored by bucket or document. Hover a point for its text.</p>
      <label class="field">
        <span class="label-text">Index</span>
        <select id="viz-index-select">
          <option value="">${indices.length ? "(choose an index)" : "no indices yet"}</option>
          ${indices.map(ix => `<option value="${escapeHtml(ix.index)}">${escapeHtml(ix.index)} (${ix["docs.count"]} docs)</option>`).join("")}
        </select>
      </label>
      <div id="viz-scatter-host" style="margin-top:1rem"></div>
    </div>
  `;

  const select = document.getElementById("viz-index-select");
  const scatterHost = document.getElementById("viz-scatter-host");
  let colorBy = "bucket";
  let lastData = null;
  // Re-render the already-fetched data under the given color-by mode (no network call).
  function switchColorMode(mode) {
    colorBy = mode;
    renderScatter(scatterHost, lastData, colorBy, switchColorMode);
  }
  select.onchange = async () => {
    if (!select.value) { scatterHost.innerHTML = ""; lastData = null; return; }
    scatterHost.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
    try {
      lastData = await api(`/api/v1/indices/${encodeURIComponent(select.value)}/vectors`);
      renderScatter(scatterHost, lastData, colorBy, switchColorMode);
    } catch (e) {
      scatterHost.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
    }
  };
}