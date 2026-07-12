// Eval page: run the retrieval evaluation harness against a golden dataset.
// Calls POST /api/v1/eval/retrieval and renders hit-rate / recall@k / MRR metrics.
import { api, errorMessage, escapeHtml, fetchIndexList, fetchIndexInfo, indexInfoHtml, wireBucketFiles, chunkTextHtml, wireChunks, toast } from "../lib.js";

function metricBarHtml(value) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const color = value >= 0.8 ? "var(--good)" : value >= 0.5 ? "var(--warn)" : "var(--bad)";
  return `<div class="score-bar-wrap"><div class="score-bar" style="width:${pct}%;background:${color}"></div></div><span class="score-num">${(value * 100).toFixed(1)}%</span>`;
}

function bucketRowHtml(bucket, m) {
  return `
    <tr>
      <td><span class="pill" style="border-color:var(--accent)">${escapeHtml(bucket)}</span></td>
      <td>${metricBarHtml(m.hit_rate ?? 0)}</td>
      <td>${metricBarHtml(m.recall ?? 0)}</td>
      <td class="mono">${(m.mrr ?? 0).toFixed(3)}</td>
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
      <div class="eval-metric"><span class="eval-metric-label">Recall</span><span class="eval-metric-val">${(c.recall * 100).toFixed(0)}%</span></div>
      <div class="eval-metric"><span class="eval-metric-label">RR</span><span class="eval-metric-val mono">${(c.reciprocal_rank ?? 0).toFixed(3)}</span></div>
      <div class="eval-metric"><span class="eval-metric-label">Retrieved</span><span class="eval-metric-val">${c.retrieved}</span></div>
      <div class="eval-metric"><span class="eval-metric-label">Matched</span><span class="eval-metric-val">${c.matched}/${c.expected}</span></div>
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

export async function render(view) {
  let indices = [];
  try { indices = await fetchIndexList(); } catch { /* fall back to empty; preview select just stays empty */ }

  view.innerHTML = `
    <h1 class="page-title">Retrieval Eval</h1>
    <p class="page-sub">Run the golden-dataset eval harness (<code class="inline">POST /api/v1/eval/retrieval</code>). Retrieval is <strong>bucket-agnostic</strong> — it searches the whole index, so make sure the chosen index actually contains the dataset's documents (a mismatch just returns 0 hits). A case's optional <code class="inline">bucket</code> only labels its metrics. Metrics: hit-rate, recall@k, MRR — overall and per label.</p>

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
      <div class="btn-row" style="margin-top:1rem">
        <button class="primary" id="ev-submit">Run eval</button>
        <span id="ev-status" class="hint"></span>
      </div>
    </div>

    <div id="ev-results"></div>
  `;

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

  document.getElementById("ev-submit").onclick = async () => {
    const status = document.getElementById("ev-status");
    const results = document.getElementById("ev-results");
    const dataset = document.getElementById("ev-dataset").value.trim();
    if (!dataset) { toast("Enter a dataset name", "error"); return; }

    const k = Number(document.getElementById("ev-k").value) || 5;
    const body = { dataset, k, top_k: k };
    // Optional index override: run every case against the selected index (empty = dataset's own)
    const indexName = document.getElementById("ev-preview-index").value;
    if (indexName) body.index_name = indexName;
    const thresholdVal = document.getElementById("ev-threshold").value;
    if (thresholdVal !== "") body.similarity_threshold = Number(thresholdVal);

    status.innerHTML = '<span class="spinner"></span> running…';
    results.innerHTML = "";
    try {
      const res = await api("/api/v1/eval/retrieval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const overall = res.overall ?? {};
      const byBucket = res.per_bucket ?? {};
      const nCases = overall.cases ?? res.cases?.length ?? "?";
      const hitCount = res.cases?.filter(c => c.matched > 0).length ?? "?";

      status.textContent = `${hitCount}/${nCases} hits`;

      results.innerHTML = `
        <div class="grid-2" style="margin-top:1rem">
          <div class="card">
            <h2 style="margin-top:0">Overall</h2>
            <div class="kv-list">
              <div class="kv-row"><span class="k">Hit-rate</span><span class="v">${metricBarHtml(overall.hit_rate ?? 0)}</span></div>
              <div class="kv-row"><span class="k">Recall@${body.k}</span><span class="v">${metricBarHtml(overall.recall ?? 0)}</span></div>
              <div class="kv-row"><span class="k">MRR</span><span class="v mono">${(overall.mrr ?? 0).toFixed(3)}</span></div>
              <div class="kv-row"><span class="k">Cases</span><span class="v">${nCases}</span></div>
            </div>
          </div>
          ${Object.keys(byBucket).length ? `
          <div class="card">
            <h2 style="margin-top:0">Per bucket</h2>
            <table class="eval-table">
              <thead><tr><th>Bucket</th><th>Hit-rate</th><th>Recall@${body.k}</th><th>MRR</th><th>n</th></tr></thead>
              <tbody>${Object.entries(byBucket).map(([b, m]) => bucketRowHtml(b, m)).join("")}</tbody>
            </table>
          </div>` : ""}
        </div>
        ${res.params ? `<div class="hint" style="margin:.5rem 0 1rem">params: ${Object.entries(res.params).filter(([,v])=>v!=null).map(([k,v])=>`${k}=${v}`).join(" · ")}</div>` : ""}
        ${res.cases?.length ? `
        <div style="margin-top:.25rem">
          <h2>Case details <span class="hint">(${res.cases.length} cases — expand each row for nodes)</span></h2>
          ${res.cases.map(c => caseCardHtml(c, body.k)).join("")}
        </div>` : ""}
      `;
      // Node bodies live in collapsed <details>; measure chunk overflow only when a row opens.
      results.querySelectorAll(".eval-details").forEach(d => {
        d.addEventListener("toggle", () => { if (d.open) wireChunks(d); });
      });
    } catch (e) {
      status.textContent = "";
      toast(errorMessage(e), "error");
      results.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
    }
  };
}
