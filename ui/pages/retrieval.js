// Retrieval page: run hybrid BM25+dense retrieval with cross-encoder rerank
// and optional metadata filters (bucket included) against a chosen index.
import { api, errorMessage, escapeHtml, toast, getKnownBuckets, fetchIndexList, indexOptionsHtml, fetchIndexInfo, indexInfoHtml, wireBucketFiles, chunkTextHtml, wireChunks, rerankerOptionsHtml, rerankerSelection } from "../lib.js";

const OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "in", "nin"];

function filterRowHtml() {
  return `
    <div class="filter-row">
      <input type="text" placeholder="key (e.g. bucket)" class="f-key">
      <select class="op">${OPERATORS.map(o => `<option>${o}</option>`).join("")}</select>
      <input type="text" placeholder="value" class="f-value">
      <button type="button" class="secondary" data-remove>×</button>
    </div>`;
}

function scoreBarHtml(score) {
  const normalized = Math.max(0, Math.min(1, (score + 5) / 15)); // rough visual scale, rerank/BM25 scores are unbounded
  return `
    <div class="score-bar-wrap"><div class="score-bar" style="width:${Math.round(normalized * 100)}%"></div></div>
    <span class="score-num">${score.toFixed(3)}</span>`;
}

function nodeCardHtml(n) {
  const metaChips = Object.entries(n.metadata || {})
    .filter(([k]) => k !== "bucket")
    .map(([k, v]) => `<span class="pill">${escapeHtml(k)}: ${escapeHtml(v)}</span>`).join("");
  const bucket = n.metadata?.bucket ? `<span class="pill" style="border-color:var(--accent)">bucket: ${escapeHtml(n.metadata.bucket)}</span>` : "";
  const rerankScore = n.score != null ? `<span class="pill" title="cross-encoder rerank score">rerank ${Number(n.score).toFixed(3)}</span>` : "";
  const retrievalScore = n.retrieval_score != null ? `<span class="pill" title="fused hybrid score (pre-rerank)">hybrid ${Number(n.retrieval_score).toFixed(3)}</span>` : "";
  return `
    <div class="result-node">
      <div class="top">${scoreBarHtml(n.score ?? 0)}</div>
      ${chunkTextHtml(n.text)}
      <div class="chip-row">${bucket}${rerankScore}${retrievalScore}${metaChips}</div>
    </div>`;
}

// One collapsible column of results for a single retrieval leg (BM25 / dense / cross-encoder).
function resultColumnHtml(title, subtitle, nodes) {
  return `
    <div class="card">
      <h3 style="margin-top:0">${escapeHtml(title)} <span class="hint">(${nodes.length})</span></h3>
      <p class="page-sub" style="margin-top:0">${escapeHtml(subtitle)}</p>
      ${nodes.length ? nodes.map(nodeCardHtml).join("") : '<div class="empty-state">No matching nodes.</div>'}
    </div>`;
}

export async function render(view) {
  let defaults = { retrieval_top_k: 10, rerank_top_n: 4 };
  let indices = [];
  try {
    const [cfg, ixs] = await Promise.all([api("/api/v1/config"), fetchIndexList()]);
    defaults = cfg;
    indices = ixs;
  } catch { /* fall back to hardcoded defaults / empty index list */ }

  const buckets = getKnownBuckets();

  view.innerHTML = `
    <h1 class="page-title">Retrieval</h1>
    <p class="page-sub">Hybrid BM25 + dense retrieval (native OpenSearch), reranked by a cross-encoder. Pick an index, inspect its documents, and test filters without invoking the LLM.</p>

    <div class="card">
      <label class="field">
        <span class="label-text">Query</span>
        <input type="text" id="r-query" placeholder="what is triage?">
      </label>
      <label class="field">
        <span class="label-text">Index</span>
        <select id="r-index">
          ${indexOptionsHtml(indices)}
        </select>
      </label>
      <div class="index-info-panel" id="r-index-info" style="display:none"></div>
      <div class="row">
        <label class="field">
          <span class="label-text">top_k</span>
          <input type="number" id="r-topk" min="1" max="100" value="${defaults.retrieval_top_k}">
        </label>
        <label class="field">
          <span class="label-text">rerank_top_n</span>
          <input type="number" id="r-rerank" min="1" max="50" value="${defaults.rerank_top_n}">
        </label>
        <label class="field">
          <span class="label-text">similarity_threshold <span class="hint">(optional override)</span></span>
          <input type="number" id="r-threshold" min="0" max="1" step="0.01" placeholder="${defaults.similarity_threshold ?? ''}">
        </label>
        <label class="field">
          <span class="label-text">Reranker</span>
          <select id="r-reranker-model">
            ${rerankerOptionsHtml(defaults)}
          </select>
        </label>
      </div>

      <label class="field">
        <span class="label-text">Metadata filters</span>
        ${buckets.length ? `<div class="chip-row" style="margin-bottom:.5rem">${buckets.map(b => `<span class="pill" data-quick-bucket="${escapeHtml(b)}" style="cursor:pointer">bucket = ${escapeHtml(b)}</span>`).join("")}</div>` : ""}
        <div id="r-filters"></div>
        <div class="btn-row">
          <button type="button" class="secondary" id="r-add-filter">+ filter</button>
          <select id="r-condition" style="max-width:110px"><option value="and">AND</option><option value="or">OR</option></select>
        </div>
      </label>

      <div class="btn-row" style="margin-top:1rem">
        <button class="primary" id="r-submit">Retrieve</button>
        <span id="r-status" class="hint"></span>
      </div>
    </div>

    <div id="r-results" class="grid-3"></div>
  `;

  // Show the selected index's recorded config + the documents it contains (📦 bucket → 📄 files)
  // so it's obvious what you're retrieving against — mirrors the Query page.
  const indexSelect = document.getElementById("r-index");
  const indexInfo = document.getElementById("r-index-info");
  async function refreshIndexInfo() {
    if (!indexSelect.value) { indexInfo.style.display = "none"; return; }
    try {
      const info = await fetchIndexInfo(indexSelect.value);
      indexInfo.innerHTML = indexInfoHtml(info, "Retrievals");
      wireBucketFiles(indexInfo);
      indexInfo.style.display = "";
    } catch { indexInfo.style.display = "none"; }
  }
  indexSelect.onchange = refreshIndexInfo;
  if (indexSelect.value) await refreshIndexInfo();

  const filtersHost = document.getElementById("r-filters");
  function addFilterRow(prefill) {
    const div = document.createElement("div");
    div.innerHTML = filterRowHtml();
    const row = div.firstElementChild;
    filtersHost.appendChild(row);
    row.querySelector("[data-remove]").onclick = () => row.remove();
    if (prefill) {
      row.querySelector(".f-key").value = prefill.key;
      row.querySelector(".f-value").value = prefill.value;
    }
  }
  document.getElementById("r-add-filter").onclick = () => addFilterRow();
  view.querySelectorAll("[data-quick-bucket]").forEach(chip => {
    chip.onclick = () => addFilterRow({ key: "bucket", value: chip.dataset.quickBucket });
  });

  document.getElementById("r-submit").onclick = async () => {
    const status = document.getElementById("r-status");
    const results = document.getElementById("r-results");
    const query = document.getElementById("r-query").value.trim();
    if (!query) { toast("Enter a query", "error"); return; }
    if (!indexSelect.value) { toast("No index selected", "error"); return; }

    const filters = [...filtersHost.querySelectorAll(".filter-row")].map(row => {
      const key = row.querySelector(".f-key").value.trim();
      if (!key) return null;
      let value = row.querySelector(".f-value").value;
      try { value = JSON.parse(value); } catch { /* keep as string */ }
      return { key, value, operator: row.querySelector(".op").value };
    }).filter(Boolean);

    const body = {
      query,
      strategy: null,
      index_name: indexSelect.value,
      top_k: Number(document.getElementById("r-topk").value) || null,
      rerank_top_n: Number(document.getElementById("r-rerank").value) || null,
      similarity_threshold: document.getElementById("r-threshold").value !== "" ? Number(document.getElementById("r-threshold").value) : null,
      filters: filters.length ? filters : null,
      filter_condition: document.getElementById("r-condition").value,
      ...rerankerSelection("r-reranker-model"),
    };

    status.innerHTML = '<span class="spinner"></span>';
    results.innerHTML = "";
    try {
      const res = await api("/api/v1/retrieve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const hybridCount = res.hybrid_nodes?.length ?? 0;
      const finalCount = res.nodes.length;
      status.textContent = `${hybridCount} post-threshold → ${finalCount} after rerank`;
      results.innerHTML = [
        resultColumnHtml("BM25", "Lexical-only hits (OpenSearch text search).", res.bm25_nodes || []),
        resultColumnHtml("Dense", "Embedding-only hits (kNN vector search).", res.dense_nodes || []),
        resultColumnHtml("Hybrid (post-threshold)", "Fused BM25+dense scores after similarity_threshold filter, before reranking.", res.hybrid_nodes || []),
        resultColumnHtml("Cross-encoder (final)", "Hybrid BM25+dense, fused then reranked — what /query uses.", res.nodes),
      ].join("");
      wireChunks(results);  // reveal Expand toggles on any chunk that overflows its clamp
    } catch (e) {
      status.textContent = "";
      toast(errorMessage(e), "error");
      results.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
    }
  };
}
