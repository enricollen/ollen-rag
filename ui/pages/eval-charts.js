// Inline-SVG charts for the eval page (no chart library; matches indices-visualizer.js). Dark-only,
// responsive viewBox, hover tooltips. Three charts: the recall/precision/nDCG @k curve, the A/B
// delta bars with bootstrap-CI whiskers, and the per-leg metric bars with rerank-lift annotation.
import { escapeHtml, CHART_PALETTE, CHART_GRID } from "../lib.js";

const AXIS = "#5c6479";   // --text-faint: recessive axis/label ink
const TEXT = "#9aa2b5";   // --text-dim: value labels
const GOOD = "#34d399";   // --good: positive delta
const BAD = "#f87171";    // --bad: negative delta
const CUTOFFS = ["1", "3", "5", "10"];

const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const signed = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(4)}`;

// Every chart lives in a positioned wrap holding the SVG and one shared tooltip div; marks carry a
// data-tip attribute that wireEvalCharts() turns into hover behaviour.
function chartWrap(inner) {
  return `<div class="chart-wrap" style="position:relative;margin-top:.4rem">${inner}
    <div class="chart-tooltip" style="display:none;position:absolute;pointer-events:none;background:var(--bg,#0d0f14);color:var(--text,#e6e9f0);border:1px solid var(--bg-raised);padding:.4rem .6rem;border-radius:6px;font-size:.82rem;white-space:nowrap;z-index:5"></div>
  </div>`;
}

// Horizontal + vertical hairline grid at the given y fractions (0..1 of the plot height).
function yGrid(x0, x1, yOf, fractions) {
  return fractions.map(f => {
    const y = yOf(f).toFixed(1);
    return `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${CHART_GRID}" stroke-width="1"></line>
            <text x="${x0 - 6}" y="${(+y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="${AXIS}">${Math.round(f * 100)}%</text>`;
  }).join("");
}

// Chart 1 — recall/precision/nDCG as k grows. Three direct-labelled lines share a 0..1 y-axis.
export function kCurveChart(overall) {
  const W = 520, H = 240, mL = 34, mR = 78, mT = 14, mB = 30;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const xOf = (i) => mL + (i / (CUTOFFS.length - 1)) * plotW;
  const yOf = (v) => mT + (1 - v) * plotH;
  const series = [
    { name: "Recall", key: "recall_at", color: CHART_PALETTE[0] },
    { name: "Precision", key: "precision_at", color: CHART_PALETTE[1] },
    { name: "nDCG", key: "ndcg_at", color: CHART_PALETTE[2] },
  ];
  const grid = yGrid(mL, W - mR, (f) => yOf(f), [0, 0.25, 0.5, 0.75, 1]);
  const xLabels = CUTOFFS.map((k, i) =>
    `<text x="${xOf(i).toFixed(1)}" y="${H - mB + 16}" text-anchor="middle" font-size="10" fill="${AXIS}">k=${k}</text>`).join("");
  const plotted = series.map(s => {
    const obj = overall[s.key] ?? {};
    const pts = CUTOFFS.map((k, i) => ({ x: xOf(i), y: yOf(obj[k] ?? 0), v: obj[k] ?? 0, k }));
    const line = `<polyline points="${pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}" fill="none" stroke="${s.color}" stroke-width="2"></polyline>`;
    const dots = pts.map(p =>
      `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4" fill="${s.color}" stroke="var(--card,#171b24)" stroke-width="2" data-tip="${s.name}@${p.k}: ${pct(p.v)}"></circle>`).join("");
    const last = pts[pts.length - 1];
    const label = `<text x="${(last.x + 8).toFixed(1)}" y="${(last.y + 3).toFixed(1)}" font-size="11" fill="${s.color}">${s.name}</text>`;
    return line + dots + label;
  }).join("");
  const legend = series.map(s =>
    `<span class="chip-row-item"><span class="legend-swatch" style="background:${s.color}"></span>${s.name}</span>`).join("");
  return chartWrap(`
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;background:var(--bg-raised);border:1px solid var(--bg-raised);border-radius:6px">
      ${grid}${xLabels}${plotted}
    </svg>`) +
    `<div class="chip-row" style="margin-top:.4rem;gap:.9rem">${legend}</div>`;
}

// Chart 2 — paired A/B deltas (B−A) as diverging bars around a zero line, each with its 95%
// bootstrap-CI whisker. A whisker clear of zero == significant (also flagged with a ●).
export function deltaChart(cmp) {
  const metrics = ["hit_rate", "recall", "mrr", "ndcg", "map"];
  const rows = metrics.filter(m => cmp.metrics?.[m]);
  const W = 560, rowH = 30, mL = 68, mR = 54, mT = 12, mB = 22;
  const H = mT + rows.length * rowH + mB;
  const plotW = W - mL - mR;
  const zeroX = mL + plotW / 2;
  // Symmetric domain covers the widest delta or CI end, floored so tiny deltas stay legible
  const D = Math.max(0.1, ...rows.flatMap(m => {
    const d = cmp.metrics[m];
    return [Math.abs(d.delta), Math.abs(d.ci[0]), Math.abs(d.ci[1])];
  }));
  const xOf = (v) => zeroX + (v / D) * (plotW / 2);
  const bars = rows.map((m, i) => {
    const d = cmp.metrics[m];
    const cy = mT + i * rowH + rowH / 2;
    const color = d.delta > 0 ? GOOD : d.delta < 0 ? BAD : AXIS;
    const x0 = Math.min(zeroX, xOf(d.delta)), x1 = Math.max(zeroX, xOf(d.delta));
    const barW = Math.max(1, x1 - x0);
    const tip = `${m} Δ ${signed(d.delta)} · 95% CI [${d.ci[0].toFixed(3)}, ${d.ci[1].toFixed(3)}]${d.significant ? " · significant" : ""}`;
    const whisker = `
      <line x1="${xOf(d.ci[0]).toFixed(1)}" y1="${cy}" x2="${xOf(d.ci[1]).toFixed(1)}" y2="${cy}" stroke="${TEXT}" stroke-width="1.5"></line>
      <line x1="${xOf(d.ci[0]).toFixed(1)}" y1="${cy - 4}" x2="${xOf(d.ci[0]).toFixed(1)}" y2="${cy + 4}" stroke="${TEXT}" stroke-width="1.5"></line>
      <line x1="${xOf(d.ci[1]).toFixed(1)}" y1="${cy - 4}" x2="${xOf(d.ci[1]).toFixed(1)}" y2="${cy + 4}" stroke="${TEXT}" stroke-width="1.5"></line>`;
    // Value label clears the whisker: right of the rightmost of (bar end, upper CI cap)
    const labelX = Math.max(x1, xOf(d.ci[1])) + 6;
    return `
      <text x="${mL - 8}" y="${cy + 3}" text-anchor="end" font-size="11" fill="${TEXT}">${m}${d.significant ? " ●" : ""}</text>
      <rect x="${x0.toFixed(1)}" y="${cy - 7}" width="${barW.toFixed(1)}" height="14" rx="3" fill="${color}" fill-opacity="0.85" data-tip="${tip}"></rect>
      ${whisker}
      <text x="${labelX.toFixed(1)}" y="${cy + 3}" font-size="10" fill="${TEXT}">${signed(d.delta)}</text>`;
  }).join("");
  return chartWrap(`
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;background:var(--bg-raised);border:1px solid var(--bg-raised);border-radius:6px">
      <line x1="${zeroX}" y1="${mT}" x2="${zeroX}" y2="${H - mB}" stroke="${AXIS}" stroke-width="1.5"></line>
      <text x="${zeroX}" y="${H - mB + 14}" text-anchor="middle" font-size="10" fill="${AXIS}">0</text>
      <text x="${mL}" y="${H - mB + 14}" text-anchor="middle" font-size="10" fill="${AXIS}">−${D.toFixed(2)}</text>
      <text x="${W - mR}" y="${H - mB + 14}" text-anchor="middle" font-size="10" fill="${AXIS}">+${D.toFixed(2)}</text>
      ${bars}
    </svg>`) +
    `<div class="chip-row" style="margin-top:.4rem;gap:.9rem">
       <span class="chip-row-item"><span class="legend-swatch" style="background:${GOOD}"></span>B better</span>
       <span class="chip-row-item"><span class="legend-swatch" style="background:${BAD}"></span>A better</span>
       <span class="chip-row-item">● = significant (CI excludes 0)</span>
     </div>`;
}

// Chart 3 — one metric across the four retrieval legs; single hue (leg identity is on the x-axis),
// with the reranked bar annotated by its lift over hybrid.
const LEGS = ["bm25", "dense", "hybrid", "reranked"];
export function legBarChart(res, metric) {
  const W = 440, H = 220, mL = 34, mR = 14, mT = 22, mB = 28;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const yOf = (v) => mT + (1 - v) * plotH;
  const slot = plotW / LEGS.length;
  const barW = slot * 0.56;
  const color = CHART_PALETTE[0];
  const grid = yGrid(mL, W - mR, (f) => yOf(f), [0, 0.25, 0.5, 0.75, 1]);
  const bars = LEGS.map((leg, i) => {
    const v = res.per_leg?.[leg]?.overall?.[metric] ?? 0;
    const x = mL + i * slot + (slot - barW) / 2;
    const y = yOf(v), h = Math.max(1, (H - mB) - y);
    return `
      <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="${color}" fill-opacity="${leg === "reranked" ? "1" : "0.7"}" data-tip="${leg} ${metric} ${v.toFixed(3)}"></rect>
      <text x="${(x + barW / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" text-anchor="middle" font-size="10" fill="${TEXT}">${v.toFixed(2)}</text>
      <text x="${(x + barW / 2).toFixed(1)}" y="${H - mB + 14}" text-anchor="middle" font-size="10" fill="${AXIS}">${leg}</text>`;
  }).join("");
  const lift = res.rerank_lift?.[metric] ?? 0;
  const liftCol = lift > 0 ? GOOD : lift < 0 ? BAD : AXIS;
  const liftLabel = `<text x="${W - mR}" y="${mT - 8}" text-anchor="end" font-size="11" fill="${liftCol}">rerank lift ${signed(lift)}</text>`;
  return chartWrap(`
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;background:var(--bg-raised);border:1px solid var(--bg-raised);border-radius:6px">
      ${grid}${liftLabel}${bars}
    </svg>`);
}

// Wire hover tooltips for every chart under `root`: each mark's data-tip fills its wrap's tooltip.
export function wireEvalCharts(root) {
  root.querySelectorAll(".chart-wrap").forEach(wrap => {
    const tooltip = wrap.querySelector(".chart-tooltip");
    if (!tooltip) return;
    const move = (e) => {
      const rect = wrap.getBoundingClientRect();
      tooltip.style.left = `${e.clientX - rect.left + 12}px`;
      tooltip.style.top = `${e.clientY - rect.top + 12}px`;
    };
    wrap.querySelectorAll("[data-tip]").forEach(mark => {
      mark.style.cursor = "pointer";
      mark.addEventListener("mouseenter", (e) => {
        tooltip.textContent = mark.dataset.tip;
        tooltip.style.display = "";
        move(e);
      });
      mark.addEventListener("mousemove", move);
      mark.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
    });
  });
}
