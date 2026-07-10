// Query (e2e) page: full RAG — retrieval + rerank + cited LLM answer.
// Inline [n] citations are clickable and jump to the matching source card.
import { api, errorMessage, escapeHtml, toast, fetchIndexList, indexOptionsHtml, fetchIndexInfo, indexInfoHtml, wireBucketFiles, chunkTextHtml, wireChunks, getQaHistory, addQaToHistory, clearQaHistory } from "../lib.js";

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

function linkCitations(answer) {
  return escapeHtml(answer).replace(/\[(\d+)\]/g, (m, n) => `<span class="cite" data-cite="${n}">[${n}]</span>`);
}

function qaItemHtml(entry) {
  const sources = (entry.sources || []).map(s => `
    <div class="source-card" data-sid="${s.id}">
      <span class="sid">[${s.id}]</span>${chunkTextHtml(s.text)}
      <div class="chip-row">
        ${s.metadata?.file_name ? `<span class="pill" style="border-color:var(--accent)">${escapeHtml(s.metadata.file_name)}</span>` : ""}
        ${s.metadata?.bucket ? `<span class="pill">bucket: ${escapeHtml(s.metadata.bucket)}</span>` : ""}
        <span class="pill">score ${Number(s.score ?? 0).toFixed(3)}</span>
      </div>
    </div>`).join("");
  return `
    <div class="qa-item">
      <div class="question">${escapeHtml(entry.query)}</div>
      <div class="answer">${linkCitations(entry.answer)}</div>
      <div class="sources-toggle"><button type="button" class="ghost" data-toggle-sources>Show ${entry.sources?.length || 0} source(s)</button></div>
      <div class="sources">${sources}</div>
    </div>`;
}

function wireQaItem(el) {
  const toggleBtn = el.querySelector("[data-toggle-sources]");
  const sourcesDiv = el.querySelector(".sources");
  toggleBtn.onclick = () => {
    sourcesDiv.classList.toggle("open");
    // Measure chunk overflow only once the sources are visible (they start display:none)
    if (sourcesDiv.classList.contains("open")) wireChunks(sourcesDiv);
    toggleBtn.textContent = (sourcesDiv.classList.contains("open") ? "Hide" : "Show") + toggleBtn.textContent.replace(/^(Show|Hide)/, "");
  };
  el.querySelectorAll(".cite").forEach(cite => {
    cite.onclick = () => {
      sourcesDiv.classList.add("open");
      wireChunks(sourcesDiv);
      toggleBtn.textContent = toggleBtn.textContent.replace(/^Show/, "Hide");
      const target = el.querySelector(`.source-card[data-sid="${cite.dataset.cite}"]`);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.style.borderLeftColor = "var(--accent-2)";
        setTimeout(() => { target.style.borderLeftColor = ""; }, 1200);
      }
    };
  });
}

export async function render(view) {
  let indices = [];
  let rerankerDefaults = { reranker_model: null, reranker_model_choices: {} };
  try { indices = await fetchIndexList(); } catch { /* fall back to empty; user can still type filters */ }
  try {
    const cfg = await api("/api/v1/config");
    rerankerDefaults = { reranker_model: cfg.reranker_model, reranker_model_choices: cfg.reranker_model_choices || {} };
  } catch { /* config unavailable, fall back silently */ }

  view.innerHTML = `
    <h1 class="page-title">Query (end-to-end)</h1>
    <p class="page-sub">Retrieval + rerank + cited LLM answer via CitationQueryEngine. Inline <code class="inline">[n]</code> citations link to the exact source chunk below.</p>

    <div class="card">
      <label class="field">
        <span class="label-text">Question</span>
        <input type="text" id="q-query" placeholder="what are the triage color codes?">
      </label>
      <div class="row">
        <label class="field">
          <span class="label-text">Index</span>
          <select id="q-index">
            ${indexOptionsHtml(indices)}
          </select>
        </label>
        <label class="field">
          <span class="label-text">Prompt name <span class="hint">(optional)</span></span>
          <input type="text" id="q-prompt" placeholder="rag_answer">
        </label>
      </div>
      <div class="index-info-panel" id="q-index-info" style="display:none"></div>
      <label class="field">
        <span class="label-text">Additional metadata filters <span class="hint">(optional)</span></span>
        <div id="q-filters"></div>
        <div class="btn-row">
          <button type="button" class="secondary" id="q-add-filter">+ filter</button>
          <select id="q-condition" style="max-width:110px"><option value="and">AND</option><option value="or">OR</option></select>
        </div>
      </label>
      <details style="margin-top:.9rem">
        <summary class="hint" style="cursor:pointer">Advanced: reranker model</summary>
        <label class="field" style="margin-top:.6rem">
          <span class="label-text">Reranker model</span>
          <select id="q-reranker-model">
            ${Object.entries(rerankerDefaults.reranker_model_choices).map(([label, value]) =>
              `<option value="${escapeHtml(value)}" ${value === rerankerDefaults.reranker_model ? "selected" : ""}>${escapeHtml(label)}</option>`
            ).join("")}
          </select>
        </label>
      </details>
      <div class="btn-row" style="margin-top:1rem">
        <button class="primary" id="q-submit">Ask</button>
        <span id="q-status" class="hint"></span>
      </div>
    </div>

    <div class="btn-row" style="justify-content:flex-end;margin-top:1rem">
      <button type="button" class="secondary" id="q-clear">Clear history</button>
    </div>
    <div class="qa-thread" id="qa-thread"></div>
  `;

  const indexSelect = document.getElementById("q-index");
  const indexInfo = document.getElementById("q-index-info");
  // Show the selected index's locked config (embedding model, chunking, buckets) so it's clear
  // the answer is generated against this index's own embedding model — no model mixing. The
  // panel's 📦 bucket picker doubles as the query's bucket filter (read at submit) — one bucket
  // selector for the page, sitting below the index dropdown.
  async function refreshIndexInfo() {
    if (!indexSelect.value) { indexInfo.style.display = "none"; return; }
    try {
      const info = await fetchIndexInfo(indexSelect.value);
      indexInfo.innerHTML = indexInfoHtml(info, "Queries");
      wireBucketFiles(indexInfo);
      indexInfo.style.display = "";
    } catch { indexInfo.style.display = "none"; }
  }
  indexSelect.onchange = refreshIndexInfo;
  if (indexSelect.value) await refreshIndexInfo();

  const filtersHost = document.getElementById("q-filters");
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
  document.getElementById("q-add-filter").onclick = () => addFilterRow();

  const thread = document.getElementById("qa-thread");
  function renderThread() {
    const history = getQaHistory();
    if (!history.length) {
      thread.innerHTML = '<div class="empty-state">Ask a question to see a cited answer here.</div>';
      return;
    }
    thread.innerHTML = history.map(qaItemHtml).join("");
    thread.querySelectorAll(".qa-item").forEach(wireQaItem);
  }
  renderThread();

  // Clear the local Q&A thread (UI-only; indexed data untouched)
  document.getElementById("q-clear").onclick = () => {
    clearQaHistory();
    renderThread();
    toast("Query history cleared", "info");
  };

  document.getElementById("q-submit").onclick = async () => {
    const status = document.getElementById("q-status");
    const query = document.getElementById("q-query").value.trim();
    if (!query) { toast("Enter a question", "error"); return; }

    const filters = [...filtersHost.querySelectorAll(".filter-row")].map(row => {
      const key = row.querySelector(".f-key").value.trim();
      if (!key) return null;
      let value = row.querySelector(".f-value").value;
      try { value = JSON.parse(value); } catch { /* keep as string */ }
      return { key, value, operator: row.querySelector(".op").value };
    }).filter(Boolean);
    // The panel's bucket picker (below the index dropdown) scopes the query to that bucket
    const selectedBucket = indexInfo.querySelector(".bucket-files-select")?.value;
    if (selectedBucket) filters.unshift({ key: "bucket", value: selectedBucket, operator: "==" });

    if (!indexSelect.value) { toast("No index selected", "error"); return; }

    const body = {
      query,
      strategy: null,
      index_name: indexSelect.value,
      prompt_name: document.getElementById("q-prompt").value.trim() || null,
      filters: filters.length ? filters : null,
      filter_condition: document.getElementById("q-condition").value,
      reranker_model: document.getElementById("q-reranker-model").value || null,
    };

    status.innerHTML = '<span class="spinner"></span> generating…';
    try {
      const res = await api("/api/v1/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      addQaToHistory({ query, answer: res.answer, sources: res.sources });
      status.textContent = "";
      document.getElementById("q-query").value = "";
      renderThread();
    } catch (e) {
      status.textContent = "";
      toast(errorMessage(e), "error");
    }
  };
}
