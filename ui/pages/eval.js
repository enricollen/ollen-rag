// Eval page: run the retrieval evaluation harness against a golden dataset.
// Calls POST /api/v1/eval/retrieval and renders hit-rate / recall@k / precision@k / MRR /
// nDCG / MAP metrics, with per-k curves, latency percentiles, and 95% bootstrap CIs.
import { api, errorMessage, escapeHtml, fetchIndexList, fetchIndexInfo, indexInfoHtml, wireBucketFiles, chunkTextHtml, wireChunks, toast, chunkingSummary } from "../lib.js";
import { kCurveChart, deltaChart, legBarChart, wireEvalCharts } from "./eval-charts.js";

// Cutoffs the backend reports curves at (mirror of evaluation.CUTOFFS)
const CUTOFFS = ["1", "3", "5", "10"];

// Plain-language glossary for every metric on the page. Each entry: what it measures + a tiny
// concrete example, so a non-specialist can read the report. Rendered into the hidden legend
// (metricsLegendHtml) and reused as hover `title=` text on metric labels via METRIC_TITLE.
const METRIC_DEFS = [
  { key: "hit_rate", name: "Hit-rate", short: "Did any correct source show up at all?",
    long: "Share of queries where at least one relevant document lands anywhere in the top-k results.",
    example: "10 queries; 8 had a correct source somewhere in their top-5 → hit-rate 80%." },
  { key: "recall", name: "Recall", short: "How many of the correct sources did we find?",
    long: "Of all documents that should have been retrieved for a query, the fraction actually returned.",
    example: "A query has 4 relevant docs; retrieval returned 3 of them → recall 0.75." },
  { key: "precision", name: "Precision@k", short: "How clean are the top-k results?",
    long: "Of the top-k returned results, the fraction that are actually relevant (the rest are noise).",
    example: "Top-5 results, 2 relevant + 3 irrelevant → precision@5 = 0.40." },
  { key: "mrr", name: "MRR", short: "How high up was the first correct hit?",
    long: "Mean Reciprocal Rank — average of 1 / (rank of the first relevant result) across queries. Higher = the right answer appears earlier.",
    example: "First correct doc at rank 2 → 1/2 = 0.5 for that query; then averaged over all queries." },
  { key: "ndcg", name: "nDCG", short: "Overall ranking quality, top-weighted.",
    long: "Normalized Discounted Cumulative Gain — rewards putting relevant docs near the top, scaled 0–1 against the ideal ordering.",
    example: "Correct docs at ranks 1 & 2 score higher than the same docs sitting at ranks 4 & 5." },
  { key: "map", name: "MAP", short: "Ranking quality across all correct sources.",
    long: "Mean Average Precision — for each query, average the precision measured at every relevant hit, then take the mean over queries. Rewards ranking all relevant docs high, not just the first.",
    example: "Relevant hits at ranks 1 and 3 → avg of P@1=1.0 and P@3=0.67 = 0.83 for that query." },
  { key: "latency", name: "Latency", short: "Time per query.",
    long: "Wall-clock retrieval time. p50 = median; p95 = slow tail (95% of queries finish faster than this).",
    example: "p50 40ms means half the queries finished in under 40ms." },
  { key: "ci", name: "95% CI", short: "How trustworthy is the number?",
    long: "95% bootstrap confidence interval — the queries are resampled many times to estimate the range the true metric likely sits in. Narrow = stable; wide = few or noisy cases.",
    example: "nDCG 0.62 with CI 0.55–0.69: on similar data, expect roughly that band." },
];

// key -> "Name — one-liner", for native hover tooltips on metric labels.
const METRIC_TITLE = Object.fromEntries(METRIC_DEFS.map(d => [d.key, `${d.name} — ${d.short}`]));

// Hidden-by-default legend: a <details> the user clicks to reveal a grid of metric explanations.
function metricsLegendHtml() {
  const items = METRIC_DEFS.map(d => `
    <div class="metric-def">
      <div class="metric-def-name">${escapeHtml(d.name)}</div>
      <div class="metric-def-long">${escapeHtml(d.long)}</div>
      <div class="metric-def-ex"><span class="metric-def-ex-tag">e.g.</span> ${escapeHtml(d.example)}</div>
    </div>`).join("");
  return `
    <details class="metric-legend">
      <summary class="metric-legend-summary">ℹ️ What do these metrics mean? <span class="hint">(click to show — or hover any metric label)</span></summary>
      <div class="metric-def-grid">${items}</div>
    </details>`;
}

function metricBarHtml(value) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const color = value >= 0.8 ? "var(--good)" : value >= 0.5 ? "var(--warn)" : "var(--bad)";
  return `<div class="score-bar-wrap"><div class="score-bar" style="width:${pct}%;background:${color}"></div></div><span class="score-num">${(value * 100).toFixed(1)}%</span>`;
}

// 95% bootstrap CI as a compact "±" hint. ci is a [lo, hi] pair in the metric's own 0-1 units;
// asPct renders it as a percentage band, otherwise as 3-decimal numbers (MRR/MAP scale).
function ciHtml(ci, asPct = true) {
  if (!Array.isArray(ci) || ci.length !== 2) return "";
  const [lo, hi] = ci;
  const fmt = asPct ? (v) => `${(v * 100).toFixed(0)}%` : (v) => v.toFixed(3);
  return `<span class="hint" style="margin-left:.4rem">95% CI ${fmt(lo)}–${fmt(hi)}</span>`;
}

// Curve table: recall@k / precision@k / nDCG@k across the reported cutoffs.
function curveTableHtml(m) {
  const cell = (obj, k) => `${((obj?.[k] ?? 0) * 100).toFixed(0)}%`;
  return `
    <table class="eval-table" style="margin-top:.4rem">
      <thead><tr><th>k</th>${CUTOFFS.map(k => `<th>${k}</th>`).join("")}</tr></thead>
      <tbody>
        <tr><td class="k">Recall@k</td>${CUTOFFS.map(k => `<td class="mono">${cell(m.recall_at, k)}</td>`).join("")}</tr>
        <tr><td class="k">Precision@k</td>${CUTOFFS.map(k => `<td class="mono">${cell(m.precision_at, k)}</td>`).join("")}</tr>
        <tr><td class="k">nDCG@k</td>${CUTOFFS.map(k => `<td class="mono">${cell(m.ndcg_at, k)}</td>`).join("")}</tr>
      </tbody>
    </table>`;
}

// Per-leg attribution: one row per retrieval leg + the reranked-vs-hybrid lift line.
const LEGS = ["bm25", "dense", "hybrid", "reranked"];
function legReportHtml(res) {
  const rows = LEGS.map(leg => {
    const o = res.per_leg?.[leg]?.overall ?? {};
    return `<tr>
      <td class="k">${leg}</td>
      <td>${metricBarHtml(o.hit_rate ?? 0)}</td>
      <td class="mono">${(o.recall ?? 0).toFixed(3)}</td>
      <td class="mono">${(o.mrr ?? 0).toFixed(3)}</td>
      <td class="mono">${(o.ndcg ?? 0).toFixed(3)}</td>
      <td class="mono">${(o.map ?? 0).toFixed(3)}</td>
    </tr>`;
  }).join("");
  const lift = res.rerank_lift ?? {};
  const liftHtml = ["ndcg", "recall", "mrr", "map", "hit_rate"].map(m => {
    const v = lift[m] ?? 0;
    const col = v > 0 ? "var(--good)" : v < 0 ? "var(--bad)" : "var(--text-dim)";
    return `<span class="pill" style="border-color:${col};color:${col}">${m} ${v >= 0 ? "+" : ""}${v.toFixed(4)}</span>`;
  }).join(" ");
  const metricOpts = ["ndcg", "recall", "mrr", "map", "hit_rate"]
    .map(m => `<option value="${m}">${m}</option>`).join("");
  return `
    <div class="card" style="margin-top:1rem">
      <h2 style="margin-top:0">Per-leg attribution</h2>
      <p class="page-sub" style="margin-top:0">Each retrieval leg scored on its own. <strong>Rerank lift</strong> = reranked − hybrid: what the cross-encoder adds. Per-leg latency is not measured (one debug call serves all legs).</p>
      <div class="row" style="align-items:center;gap:.5rem;margin-bottom:.2rem">
        <span class="hint">Chart metric</span>
        <select id="ev-leg-metric" style="max-width:9rem">${metricOpts}</select>
      </div>
      <div id="ev-leg-chart">${legBarChart(res, "ndcg")}</div>
      <table class="eval-table" style="margin-top:.6rem">
        <thead><tr><th>Leg</th><th>Hit-rate</th><th>Recall</th><th>MRR</th><th>nDCG</th><th>MAP</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="eval-section-label" style="margin-top:.7rem">Rerank lift</div>
      <div>${liftHtml}</div>
    </div>`;
}

// Paired A/B comparison table: mean delta (B − A) with bootstrap CI and significance star.
function compareHtml(cmp, labelA, labelB) {
  const rows = Object.entries(cmp.metrics ?? {}).map(([m, d]) => {
    const col = d.delta > 0 ? "var(--good)" : d.delta < 0 ? "var(--bad)" : "var(--text-dim)";
    return `<tr>
      <td class="k">${m}</td>
      <td class="mono" style="color:${col}">${d.delta >= 0 ? "+" : ""}${d.delta.toFixed(4)}</td>
      <td class="mono">[${d.ci[0].toFixed(3)}, ${d.ci[1].toFixed(3)}]</td>
      <td>${d.significant ? `<span class="pill pill-ok">significant</span>` : `<span class="hint">n.s.</span>`}</td>
    </tr>`;
  }).join("");
  return `
    <div class="card" style="margin-top:1rem">
      <h2 style="margin-top:0">A/B comparison <span class="hint">(${cmp.n_paired} paired cases)</span></h2>
      <p class="page-sub" style="margin-top:0">Delta = <strong>${escapeHtml(labelB)}</strong> − <strong>${escapeHtml(labelA)}</strong>, paired by query. 95% bootstrap CI; a metric is <em>significant</em> when its CI excludes 0.</p>
      <table class="eval-table">
        <thead><tr><th>Metric</th><th>Δ (B−A)</th><th>95% CI</th><th></th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${(cmp.a_only?.length || cmp.b_only?.length) ? `<div class="hint" style="margin-top:.5rem">unpaired: ${cmp.a_only?.length ?? 0} only in A, ${cmp.b_only?.length ?? 0} only in B (excluded)</div>` : ""}
    </div>`;
}

// Compact "which system" tag for a saved run: vector store + each resolved index's embedding
// model (index name only when it differs from a dataset run over a single index). Runs saved
// before this field existed have no params.system — falls back silently.
function systemTagText(system) {
  if (!system) return "";
  const indices = Object.entries(system.indices || {});
  if (!indices.length) return system.vector_store ? ` · ${system.vector_store}` : "";
  const bits = indices.map(([idx, m]) => {
    const emb = m.embedding_provider ? `${m.embedding_provider}/${m.embedding_model}` : "?";
    return indices.length > 1 ? `${idx}:${emb}` : emb;
  });
  return ` · ${system.vector_store}/${bits.join("+")}`;
}

// One <option> for the run picker: timestamp + label + headline nDCG + system (store/embedding)
function runOptionHtml(run) {
  const nd = run.overall?.ndcg != null ? ` · nDCG ${run.overall.ndcg.toFixed(3)}` : "";
  const lbl = run.label ? ` · ${run.label}` : "";
  const sys = systemTagText(run.params?.system);
  return `<option value="${escapeHtml(run.id)}">${escapeHtml(run.timestamp ?? run.id)}${escapeHtml(lbl)}${nd}${escapeHtml(sys)}</option>`;
}

// Headline metrics + system tag for a picked run, shown under its A/B select so the two sides are
// legible at a glance without opening the compare. Falls back quietly for older runs.
function runMetaHtml(run) {
  if (!run) return `<span class="hint">no run selected</span>`;
  const o = run.overall ?? {};
  const bits = [];
  if (o.ndcg != null) bits.push(`nDCG ${o.ndcg.toFixed(3)}`);
  if (o.recall != null) bits.push(`recall ${o.recall.toFixed(3)}`);
  if (o.hit_rate != null) bits.push(`hit ${(o.hit_rate * 100).toFixed(0)}%`);
  const sys = systemTagText(run.params?.system).replace(/^ · /, "");
  return `<span class="mono">${bits.join(" · ") || "—"}</span>${sys ? `<div class="hint">${escapeHtml(sys)}</div>` : ""}`;
}

// System block above the metric delta table: which vector store/embedding/chunking each run
// actually used, so a compare is legible as "system X vs system Y", not just abstract deltas.
function systemRowHtml(run, letter) {
  const sys = run?.params?.system;
  if (!sys) return `<div class="kv-row"><span class="k">${letter}</span><span class="v hint">system config not recorded (older run)</span></div>`;
  const indices = Object.entries(sys.indices || {}).map(([idx, m]) => `
    <div class="hint" style="margin-left:1rem">
      <code class="inline">${escapeHtml(idx)}</code> —
      ${escapeHtml(m.embedding_provider ? `${m.embedding_provider}/${m.embedding_model}` : "unrecorded")} ·
      ${escapeHtml(chunkingSummary(m.chunking))}
    </div>`).join("");
  return `<div class="kv-row"><span class="k">${letter}</span><span class="v">🗄️ ${escapeHtml(sys.vector_store ?? "?")}</span></div>${indices}`;
}
function systemCompareHtml(runA, runB) {
  return `
    <div class="card" style="margin-top:1rem">
      <h2 style="margin-top:0">System &amp; params</h2>
      <p class="page-sub" style="margin-top:0">What each run actually hit — so a metric delta is legible as one system vs another, not just numbers.</p>
      <div class="kv-list">${systemRowHtml(runA, "A")}</div>
      <div class="kv-list" style="margin-top:.5rem">${systemRowHtml(runB, "B")}</div>
    </div>`;
}

function bucketRowHtml(bucket, m) {
  return `
    <tr>
      <td><span class="pill" style="border-color:var(--accent)">${escapeHtml(bucket)}</span></td>
      <td>${metricBarHtml(m.hit_rate ?? 0)}</td>
      <td>${metricBarHtml(m.recall ?? 0)}</td>
      <td class="mono">${(m.mrr ?? 0).toFixed(3)}</td>
      <td class="mono">${(m.ndcg ?? 0).toFixed(3)}</td>
      <td class="mono">${(m.map ?? 0).toFixed(3)}</td>
      <td class="mono">${(m.latency_ms?.p50 ?? 0).toFixed(0)}ms</td>
      <td>${m.cases ?? "—"}</td>
    </tr>`;
}

function caseCardHtml(c, k) {
  const hit = c.matched > 0;
  const statusBadge = hit
    ? `<span class="eval-badge eval-badge-hit">✓ HIT rank ${c.first_rank}</span>`
    : `<span class="eval-badge eval-badge-miss">✗ MISS</span>`;

  const metricsRow = `
    <div class="eval-metrics-row">
      <div class="eval-metric" title="${escapeHtml(METRIC_TITLE.recall)}"><span class="eval-metric-label">Recall</span><span class="eval-metric-val">${(c.recall * 100).toFixed(0)}%</span></div>
      <div class="eval-metric" title="${escapeHtml(METRIC_TITLE.precision)}"><span class="eval-metric-label">P@5</span><span class="eval-metric-val">${((c.precision_at?.["5"] ?? 0) * 100).toFixed(0)}%</span></div>
      <div class="eval-metric" title="Reciprocal rank — 1 / rank of this query's first correct hit (rank 2 → 0.5)."><span class="eval-metric-label">RR</span><span class="eval-metric-val mono">${(c.reciprocal_rank ?? 0).toFixed(3)}</span></div>
      <div class="eval-metric" title="${escapeHtml(METRIC_TITLE.ndcg)}"><span class="eval-metric-label">nDCG</span><span class="eval-metric-val mono">${(c.ndcg ?? 0).toFixed(3)}</span></div>
      <div class="eval-metric" title="Average precision — this query's contribution to MAP (precision averaged at each relevant hit)."><span class="eval-metric-label">AP</span><span class="eval-metric-val mono">${(c.average_precision ?? 0).toFixed(3)}</span></div>
      <div class="eval-metric" title="${escapeHtml(METRIC_TITLE.latency)}"><span class="eval-metric-label">Latency</span><span class="eval-metric-val mono">${(c.latency_ms ?? 0).toFixed(0)}ms</span></div>
      <div class="eval-metric" title="Relevant sources found for this query vs how many were expected."><span class="eval-metric-label">Matched</span><span class="eval-metric-val">${c.matched}/${c.expected}</span></div>
    </div>`;

  const expectedHtml = (c.expected_chunks ?? []).map(e => `
    <div class="eval-expected-chunk">
      <span class="pill" style="border-color:var(--accent-2)">${escapeHtml(e.file_name)}</span>
      ${e.contains ? `<span class="eval-contains">"${escapeHtml(e.contains)}"</span>` : ""}
    </div>`).join("");

  const nodesHtml = (c.retrieved_nodes ?? []).map(n => `
    <div class="eval-node ${n.matched ? "eval-node-matched" : ""}">
      <div class="eval-node-header">
        <span class="eval-node-rank">#${n.rank}</span>
        <span class="pill ${n.matched ? "pill-ok" : ""}">${escapeHtml(n.file_name)}</span>
        ${n.score != null ? `<span class="mono" style="font-size:.75rem;color:var(--text-dim)">score ${n.score}</span>` : ""}
        ${n.matched ? `<span class="eval-badge eval-badge-hit" style="padding:.1rem .5rem;font-size:.7rem">match</span>` : ""}
      </div>
      ${chunkTextHtml(n.text, "eval-node-text")}
    </div>`).join("");

  return `
    <div class="eval-case-card ${hit ? "eval-case-hit" : "eval-case-miss"}">
      <div class="eval-case-header">
        <div class="eval-case-query">${escapeHtml(c.query)}</div>
        <div style="display:flex;align-items:center;gap:.6rem;flex-shrink:0">
          ${c.bucket ? `<span class="pill" style="border-color:var(--accent)">bucket: ${escapeHtml(c.bucket)}</span>` : ""}
          ${statusBadge}
        </div>
      </div>
      ${metricsRow}
      <details class="eval-details">
        <summary class="eval-details-summary">Expected sources &amp; retrieved nodes</summary>
        <div class="eval-details-body">
          <div class="eval-section-label">Expected (${c.expected_chunks?.length ?? c.expected})</div>
          <div class="eval-expected">${expectedHtml || "<em>—</em>"}</div>
          <div class="eval-section-label" style="margin-top:.75rem">Retrieved nodes (${c.retrieved_nodes?.length ?? c.retrieved})</div>
          <div>${nodesHtml || "<em>No nodes returned</em>"}</div>
        </div>
      </details>
    </div>`;
}

// Render a whole-pipeline eval report (overall + per-bucket + per-case) into the results element.
function renderReport(res, k, results) {
  const overall = res.overall ?? {};
  const byBucket = res.per_bucket ?? {};
  const nCases = overall.cases ?? res.cases?.length ?? "?";
  const nHit = res.cases?.filter(c => c.matched > 0).length ?? 0;
  const nMiss = (res.cases?.length ?? 0) - nHit;
  const ci = overall.ci ?? {};
  const lat = overall.latency_ms ?? {};
  results.innerHTML = `
    <div class="grid-2" style="margin-top:1rem">
      <div class="card">
        <h2 style="margin-top:0">Overall</h2>
        <div class="kv-list">
          <div class="kv-row" title="${escapeHtml(METRIC_TITLE.hit_rate)}"><span class="k">Hit-rate</span><span class="v">${metricBarHtml(overall.hit_rate ?? 0)}${ciHtml(ci.hit_rate)}</span></div>
          <div class="kv-row" title="${escapeHtml(METRIC_TITLE.recall)}"><span class="k">Recall (all)</span><span class="v">${metricBarHtml(overall.recall ?? 0)}${ciHtml(ci.recall)}</span></div>
          <div class="kv-row" title="${escapeHtml(METRIC_TITLE.ndcg)}"><span class="k">nDCG@10</span><span class="v">${metricBarHtml(overall.ndcg ?? 0)}${ciHtml(ci.ndcg)}</span></div>
          <div class="kv-row" title="${escapeHtml(METRIC_TITLE.map)}"><span class="k">MAP</span><span class="v">${metricBarHtml(overall.map ?? 0)}${ciHtml(ci.map)}</span></div>
          <div class="kv-row" title="${escapeHtml(METRIC_TITLE.mrr)}"><span class="k">MRR</span><span class="v mono">${(overall.mrr ?? 0).toFixed(3)}${ciHtml(ci.mrr, false)}</span></div>
          <div class="kv-row" title="${escapeHtml(METRIC_TITLE.latency)}"><span class="k">Latency</span><span class="v mono">p50 ${(lat.p50 ?? 0).toFixed(0)}ms · p95 ${(lat.p95 ?? 0).toFixed(0)}ms</span></div>
          <div class="kv-row"><span class="k">Cases</span><span class="v">${nCases}</span></div>
        </div>
        <div class="eval-section-label" style="margin-top:.6rem">Curves</div>
        ${kCurveChart(overall)}
        ${curveTableHtml(overall)}
      </div>
      ${Object.keys(byBucket).length ? `
      <div class="card">
        <h2 style="margin-top:0">Per bucket</h2>
        <table class="eval-table">
          <thead><tr><th>Bucket</th><th>Hit-rate</th><th>Recall</th><th>MRR</th><th>nDCG</th><th>MAP</th><th>p50</th><th>n</th></tr></thead>
          <tbody>${Object.entries(byBucket).map(([b, m]) => bucketRowHtml(b, m)).join("")}</tbody>
        </table>
      </div>` : ""}
    </div>
    ${res.run_id ? `<div class="hint" style="margin:.5rem 0">saved as run <code class="inline">${escapeHtml(res.run_id)}</code></div>` : ""}
    ${res.params ? `<div class="hint" style="margin:.5rem 0 1rem">params: ${Object.entries(res.params).filter(([k,v])=>v!=null && k !== "system").map(([pk,pv])=>`${pk}=${pv}`).join(" · ")}${escapeHtml(systemTagText(res.params.system))}</div>` : ""}
    ${res.cases?.length ? `
    <details class="eval-cases-fold" style="margin-top:.5rem">
      <summary class="eval-cases-fold-summary">
        <span class="eval-cases-fold-title">Case details</span>
        <span class="eval-cases-fold-meta">${res.cases.length} cases · <span style="color:var(--good)">${nHit} hit</span> / <span style="color:var(--bad)">${nMiss} miss</span></span>
        <span class="hint eval-cases-fold-cta">click to expand</span>
      </summary>
      <div class="eval-cases-toolbar">
        <span class="hint">Per-query breakdown — expected sources vs what retrieval actually returned.</span>
        <span class="spacer"></span>
        <button class="ghost" id="ev-expand-all">Expand all nodes</button>
        <button class="ghost" id="ev-collapse-all">Collapse all</button>
      </div>
      <div class="eval-cases-list">
        ${res.cases.map(c => caseCardHtml(c, k)).join("")}
      </div>
    </details>` : ""}
  `;
  // Node bodies live in collapsed <details>; measure chunk overflow only when a row opens.
  results.querySelectorAll(".eval-details").forEach(d => {
    d.addEventListener("toggle", () => { if (d.open) wireChunks(d); });
  });
  // Expand/collapse every per-case node panel at once. Expanding wires chunk overflow up front.
  const expandAll = results.querySelector("#ev-expand-all");
  const collapseAll = results.querySelector("#ev-collapse-all");
  if (expandAll) expandAll.onclick = () => {
    results.querySelectorAll(".eval-cases-list .eval-details").forEach(d => { d.open = true; wireChunks(d); });
  };
  if (collapseAll) collapseAll.onclick = () => {
    results.querySelectorAll(".eval-cases-list .eval-details").forEach(d => { d.open = false; });
  };
  wireEvalCharts(results);
}

export async function render(view) {
  let indices = [];
  try { indices = await fetchIndexList(); } catch { /* fall back to empty; preview select just stays empty */ }

  view.innerHTML = `
    <h1 class="page-title">Retrieval Eval</h1>
    <p class="page-sub">Run the golden-dataset eval harness (<code class="inline">POST /api/v1/eval/retrieval</code>). Retrieval is <strong>bucket-agnostic</strong> — it searches the whole index, so make sure the chosen index actually contains the dataset's documents (a mismatch just returns 0 hits). A case's optional <code class="inline">bucket</code> only labels its metrics. Metrics: hit-rate, recall@k, precision@k, MRR, nDCG, MAP — with per-k curves, latency, and 95% bootstrap CIs, overall and per label.</p>

    ${metricsLegendHtml()}

    <div class="card">
      <label class="field">
        <span class="label-text">Index <span class="hint">(all cases run against this index, using its locked embedding model; leave empty to use each case's own index from the dataset)</span></span>
        <select id="ev-preview-index">
          <option value="">(use dataset's per-case index)</option>
          ${indices.map(ix => `<option value="${escapeHtml(ix.index)}">${escapeHtml(ix.index)} (${ix["docs.count"]} docs)</option>`).join("")}
        </select>
      </label>
      <div class="index-info-panel" id="ev-preview-info" style="display:none"></div>
      <label class="field">
        <span class="label-text">Dataset name <span class="hint">(stem only, e.g. <code class="inline">example_bucket</code> → <code class="inline">config/eval/example_bucket.yaml</code>)</span></span>
        <input type="text" id="ev-dataset" placeholder="example_bucket" value="example_bucket">
      </label>
      <div class="row">
        <label class="field">
          <span class="label-text">k (recall@k)</span>
          <input type="number" id="ev-k" min="1" max="50" value="5">
        </label>
        <label class="field">
          <span class="label-text">similarity_threshold <span class="hint">(optional override)</span></span>
          <input type="number" id="ev-threshold" min="0" max="1" step="0.01" placeholder="from settings">
        </label>
      </div>
      <div class="row" style="margin-top:.5rem;align-items:center;gap:1.2rem;flex-wrap:wrap">
        <label class="checkbox-inline"><input type="checkbox" id="ev-per-leg"> Per-leg attribution <span class="hint">(bm25/dense/hybrid/reranked + rerank lift)</span></label>
        <label class="checkbox-inline"><input type="checkbox" id="ev-save"> Save run <span class="hint" id="ev-save-hint" style="display:none">(not available with per-leg attribution)</span></label>
        <input type="text" id="ev-label" placeholder="run label (optional)" style="max-width:16rem">
      </div>
      <div class="btn-row" style="margin-top:1rem">
        <button class="primary" id="ev-submit">Run eval</button>
        <span id="ev-status" class="hint"></span>
      </div>
    </div>

    <div id="ev-results"></div>

    <div class="card" id="ev-history-card" style="margin-top:1rem">
      <h2 style="margin-top:0">Run history &amp; A/B compare</h2>
      <p class="page-sub" style="margin-top:0">Every eval you save (tick <code class="inline">Save run</code> above) is stored here. Pick two and hit compare to see whether a change — different embedding model, chunking, threshold — actually moved the numbers. Runs are matched <strong>per query</strong>, so the delta is a fair like-for-like, and each metric is flagged <em>significant</em> when its confidence interval clears zero (i.e. unlikely to be luck).</p>
      <div class="ab-compare">
        <div class="ab-side ab-side-a">
          <div class="ab-side-label">A · Baseline <span class="hint">what you measure against</span></div>
          <select id="ev-run-a"></select>
          <div class="ab-side-meta" id="ev-run-a-meta"></div>
        </div>
        <div class="ab-arrow" aria-hidden="true">→</div>
        <div class="ab-side ab-side-b">
          <div class="ab-side-label">B · Variant <span class="hint">the change you're testing</span></div>
          <select id="ev-run-b"></select>
          <div class="ab-side-meta" id="ev-run-b-meta"></div>
        </div>
      </div>
      <div class="btn-row" style="margin-top:.9rem;align-items:center">
        <button class="primary" id="ev-compare">Compare A → B</button>
        <button id="ev-refresh-runs" class="ghost">Refresh list</button>
        <span class="hint" id="ev-history-empty" style="display:none">No saved runs yet — tick “Save run” above, run an eval, then come back.</span>
      </div>
      <div id="ev-compare-result"></div>
    </div>
  `;

  // Per-leg reports have no top-level "cases" list shaped like a normal report, so save_run
  // can't persist them (compare_runs pairs cases by query) — the backend just drops `save`
  // silently (routes.py per_leg branch returns before save_run is ever called). Disable the
  // checkbox instead of letting the user check both and get no run, no error, no explanation.
  const perLegBox = document.getElementById("ev-per-leg");
  const saveBox = document.getElementById("ev-save");
  const saveHint = document.getElementById("ev-save-hint");
  perLegBox.onchange = () => {
    saveBox.disabled = perLegBox.checked;
    if (perLegBox.checked) saveBox.checked = false;
    saveHint.style.display = perLegBox.checked ? "" : "none";
  };

  const previewSelect = document.getElementById("ev-preview-index");
  const previewInfo = document.getElementById("ev-preview-info");
  // Show the full locked config (embedding model, chunking, buckets) of the previewed index so
  // it's clear eval never mixes models — each case runs with its dataset index's own model.
  previewSelect.onchange = async () => {
    if (!previewSelect.value) { previewInfo.style.display = "none"; return; }
    previewInfo.innerHTML = '<span class="spinner"></span> loading…';
    previewInfo.style.display = "";
    try {
      const info = await fetchIndexInfo(previewSelect.value);
      previewInfo.innerHTML = indexInfoHtml(info, "Eval cases");
      wireBucketFiles(previewInfo);
    } catch (e) {
      previewInfo.style.display = "none";
      toast(errorMessage(e), "error");
    }
  };

  // --- Run history & A/B compare ---
  const runA = document.getElementById("ev-run-a");
  const runB = document.getElementById("ev-run-b");
  const compareResult = document.getElementById("ev-compare-result");
  let runsById = {};  // id -> run summary (incl. params.system), so compare can show "which system" without a refetch

  // Refresh the headline-metric line under each picker whenever a selection changes.
  function refreshMeta() {
    document.getElementById("ev-run-a-meta").innerHTML = runMetaHtml(runsById[runA.value]);
    document.getElementById("ev-run-b-meta").innerHTML = runMetaHtml(runsById[runB.value]);
  }
  runA.onchange = refreshMeta;
  runB.onchange = refreshMeta;

  async function loadRuns() {
    try {
      const { runs } = await api("/api/v1/eval/runs");
      runsById = Object.fromEntries(runs.map(r => [r.id, r]));
      const opts = runs.map(runOptionHtml).join("");
      // Keep current selections if still present; default B to the newest, A to the next
      const prevA = runA.value, prevB = runB.value;
      runA.innerHTML = opts;
      runB.innerHTML = opts;
      if (runs.length) {
        runA.value = prevA || (runs[1]?.id ?? runs[0].id);
        runB.value = prevB || runs[0].id;
      }
      document.getElementById("ev-history-card").style.opacity = runs.length ? "1" : ".7";
      document.getElementById("ev-history-empty").style.display = runs.length ? "none" : "";
      refreshMeta();
    } catch { /* history is best-effort; leave selects empty */ }
  }

  document.getElementById("ev-refresh-runs").onclick = loadRuns;

  document.getElementById("ev-compare").onclick = async () => {
    if (!runA.value || !runB.value) { toast("Save at least two runs to compare", "error"); return; }
    if (runA.value === runB.value) { toast("Pick two different runs", "error"); return; }
    compareResult.innerHTML = '<div class="hint"><span class="spinner"></span> comparing…</div>';
    try {
      const cmp = await api("/api/v1/eval/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ a: runA.value, b: runB.value }),
      });
      const labelA = runA.options[runA.selectedIndex].text;
      const labelB = runB.options[runB.selectedIndex].text;
      compareResult.innerHTML = systemCompareHtml(runsById[runA.value], runsById[runB.value])
        + compareHtml(cmp, labelA, labelB) + deltaChart(cmp);
      wireEvalCharts(compareResult);
    } catch (e) {
      compareResult.innerHTML = "";
      toast(errorMessage(e), "error");
    }
  };

  loadRuns();

  document.getElementById("ev-submit").onclick = async () => {
    const status = document.getElementById("ev-status");
    const results = document.getElementById("ev-results");
    const dataset = document.getElementById("ev-dataset").value.trim();
    if (!dataset) { toast("Enter a dataset name", "error"); return; }

    const k = Number(document.getElementById("ev-k").value) || 5;
    const perLeg = document.getElementById("ev-per-leg").checked;
    const save = document.getElementById("ev-save").checked;
    const label = document.getElementById("ev-label").value.trim();
    const body = { dataset, k, top_k: k };
    // Optional index override: run every case against the selected index (empty = dataset's own)
    const indexName = document.getElementById("ev-preview-index").value;
    if (indexName) body.index_name = indexName;
    const thresholdVal = document.getElementById("ev-threshold").value;
    if (thresholdVal !== "") body.similarity_threshold = Number(thresholdVal);
    if (perLeg) body.per_leg = true;
    if (save && !perLeg) { body.save = true; if (label) body.label = label; }

    status.innerHTML = '<span class="spinner"></span> running…';
    results.innerHTML = "";
    try {
      const res = await api("/api/v1/eval/retrieval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      // Per-leg attribution comes back in its own shape (per_leg + rerank_lift)
      if (perLeg) {
        status.textContent = "per-leg done";
        results.innerHTML = legReportHtml(res);
        wireEvalCharts(results);
        // Metric selector re-renders just the leg bar chart, then re-wires its tooltips
        const legMetric = document.getElementById("ev-leg-metric");
        const legChart = document.getElementById("ev-leg-chart");
        legMetric.onchange = () => {
          legChart.innerHTML = legBarChart(res, legMetric.value);
          wireEvalCharts(legChart);
        };
        return;
      }

      const overall = res.overall ?? {};
      const nCases = overall.cases ?? res.cases?.length ?? "?";
      const hitCount = res.cases?.filter(c => c.matched > 0).length ?? "?";
      status.textContent = `${hitCount}/${nCases} hits`;
      renderReport(res, k, results);
      if (res.run_id) { toast(`Saved run ${res.run_id}`, "ok"); loadRuns(); }
    } catch (e) {
      status.textContent = "";
      toast(errorMessage(e), "error");
      results.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
    }
  };
}
