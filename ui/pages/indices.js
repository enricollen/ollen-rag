// Indices page: two sub-tabs. "Explore existing" — the cross-store index picker doubles as the
// browser: click an index (in the active store) to browse its buckets and raw stored chunks, or 🗑
// to permanently delete it (typed confirmation; inactive-store indexes are read-only). "Create new
// index" — build a brand-new index from scratch (name, chunking, embedding, files); moved here from
// the Ingestion KB page.
import { api, errorMessage, escapeHtml, toast, chunkTextHtml, wireChunks, fetchIndexInfo, indexOverviewHtml, wireIndicesOverview, getKnownBuckets, addJobToHistory, clearJobHistory } from "../lib.js";
import { STRATEGIES, CHUNK_FIELDS, strategyCardHtml, chunkInputsHtml, renderJobHistory, addMetaRow, wireDropzone, runIngestBatch } from "../ingest-common.js";
import { buildVisualizerTab } from "./indices-visualizer.js";

const PAGE_SIZE = 20;
let currentPage = 0;
let currentIndex = null;
let currentBucket = null;  // when set, the "stored chunks" pager is scoped to this bucket

function docCardHtml(doc) {
  const chips = Object.entries(doc.metadata || {})
    .filter(([k]) => !["_node_type", "_node_content"].includes(k))
    .map(([k, v]) => `<span class="pill">${escapeHtml(k)}: ${escapeHtml(String(v).slice(0, 60))}</span>`).join("");
  return `
    <div class="result-node">
      <div class="hint" style="margin-bottom:.4rem">id: <code class="inline">${escapeHtml(doc.id)}</code></div>
      ${chunkTextHtml(doc.content)}
      <div class="chip-row">${chips}</div>
    </div>`;
}

// One bucket entry: the clickable 📦 card (select-to-filter) plus a 🗑 delete button.
// Wrapped in a container because a <button> cannot nest inside the card <button>.
function bucketCardHtml(name, count) {
  return `
    <div class="bucket-card-wrap">
      <button type="button" class="bucket-card" data-bucket="${escapeHtml(name)}">
        <span class="bucket-card-icon">📦</span>
        <span class="bucket-card-name">${escapeHtml(name)}</span>
        <span class="bucket-card-count">${count} doc${count === 1 ? "" : "s"}</span>
      </button>
      <button type="button" class="bucket-del" data-bucket="${escapeHtml(name)}" data-count="${count}" title="Delete this bucket">🗑 Delete</button>
    </div>`;
}

// Fetch the selected index's info and render its buckets as cards. Clicking a card lists that
// bucket's 📄 file names AND refreshes the "stored chunks" pager below, scoped to that bucket;
// clicking the selected card again clears the scope (back to all chunks in the index).
async function loadBuckets(view, indexName) {
  const host = document.getElementById("buckets-host");
  const docsHost = document.getElementById("bucket-docs-host");
  docsHost.innerHTML = "";
  host.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  try {
    const info = await fetchIndexInfo(indexName);
    const bucketFiles = info.bucket_files || {};
    const names = Object.keys(bucketFiles);
    if (!names.length) { host.innerHTML = '<div class="empty-state">No buckets in this index.</div>'; return; }
    host.innerHTML = names.map(n => bucketCardHtml(n, bucketFiles[n].length)).join("");
    host.querySelectorAll(".bucket-card").forEach(card => {
      card.onclick = () => {
        const bucket = card.dataset.bucket;
        const alreadySelected = card.classList.contains("selected");
        // Toggle: re-clicking the active bucket clears the filter; otherwise select this one.
        currentBucket = alreadySelected ? null : bucket;
        host.querySelectorAll(".bucket-card").forEach(c => c.classList.toggle("selected", !alreadySelected && c === card));
        if (currentBucket) {
          const files = bucketFiles[bucket] || [];
          docsHost.innerHTML = `
            <div class="bucket-docs-title">📦 ${escapeHtml(bucket)} — ${files.length} document${files.length === 1 ? "" : "s"}</div>
            <ul class="bucket-docs-list">${files.map(f => `<li class="bucket-doc-item">📄 ${escapeHtml(f)}</li>`).join("") || '<li class="hint">empty</li>'}</ul>`;
        } else {
          docsHost.innerHTML = "";
        }
        // Refresh the chunks list to reflect the index + bucket selection.
        currentPage = 0;
        loadDocuments(view, currentIndex, currentPage);
      };
    });
    // Wire the 🗑 delete button on each bucket card: confirm, DELETE, then refresh
    // the buckets, stored chunks, and the overview doc counts.
    host.querySelectorAll(".bucket-del").forEach(btn => {
      btn.onclick = async (e) => {
        e.stopPropagation();  // don't trigger the card's select-to-filter click
        const bucket = btn.dataset.bucket;
        const count = btn.dataset.count;
        if (!confirm(`Delete bucket "${bucket}" and its ${count} document(s) from "${indexName}"? This cannot be undone.`)) return;
        try {
          const res = await api(`/api/v1/indices/${encodeURIComponent(indexName)}/buckets/${encodeURIComponent(bucket)}`, { method: "DELETE" });
          toast(`Deleted bucket "${bucket}" (${res.deleted} document(s))`, "success");
          if (currentBucket === bucket) currentBucket = null;  // clear a stale chunk filter
          currentPage = 0;
          await loadBuckets(view, indexName);                    // re-render bucket cards
          await loadDocuments(view, currentIndex, currentPage);  // refresh stored chunks
          refreshOverview(view);                                 // refresh overview doc counts
        } catch (err) {
          toast(errorMessage(err), "error");
        }
      };
    });
  } catch (e) {
    host.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
  }
}

async function loadDocuments(view, indexName, page) {
  const docsHost = document.getElementById("docs-host");
  const pager = document.getElementById("pager-info");
  docsHost.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  try {
    const bucketQs = currentBucket ? `&bucket=${encodeURIComponent(currentBucket)}` : "";
    const res = await api(`/api/v1/indices/${encodeURIComponent(indexName)}/documents?offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}${bucketQs}`);
    if (!res.documents.length) {
      docsHost.innerHTML = `<div class="empty-state">No documents${currentBucket ? ` in bucket "${escapeHtml(currentBucket)}"` : " in this index"}.</div>`;
    } else {
      docsHost.innerHTML = res.documents.map(docCardHtml).join("");
      wireChunks(docsHost);  // reveal Expand toggles on any document body that overflows its clamp
    }
    const totalPages = Math.max(1, Math.ceil(res.total / PAGE_SIZE));
    const scope = currentBucket ? ` · bucket "${currentBucket}"` : "";
    pager.textContent = `page ${page + 1} / ${totalPages} · ${res.total} document(s) total${scope}`;
    document.getElementById("prev-page").disabled = page === 0;
    document.getElementById("next-page").disabled = page + 1 >= totalPages;
  } catch (e) {
    docsHost.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
    pager.textContent = "";
  }
}

// Re-fetch the cross-store overview and (re)render the picker panel + its selection/delete wiring.
async function refreshOverview(view) {
  const host = document.getElementById("overview-host");
  host.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  let overview = null;
  try { overview = await api("/api/v1/indices/overview"); } catch (e) { host.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`; return; }
  host.innerHTML = indexOverviewHtml(overview, { deletable: true, selectable: true });
  // Keep the previously-selected card highlighted across refreshes when it still exists.
  const stillExists = overview.stores.some(s => s.active && s.indices.some(i => i.index === currentIndex));
  if (!stillExists) currentIndex = null;
  markSelectedCard(host);
  wireIndicesOverview(host, overview, {
    onSelect: (store, index) => selectIndex(view, host, index),
    onDelete: (store, index) => deleteIndex(view, index),
  });
  if (currentIndex) { loadBuckets(view, currentIndex); loadDocuments(view, currentIndex, currentPage); }
  else showBrowsePlaceholder();
}

// Highlight the card matching currentIndex (active store only).
function markSelectedCard(host) {
  host.querySelectorAll(".kb-index").forEach(c => c.classList.toggle("selected", c.dataset.index === currentIndex));
}

function selectIndex(view, host, index) {
  currentIndex = index;
  currentPage = 0;
  currentBucket = null;
  markSelectedCard(host);
  loadBuckets(view, currentIndex);
  loadDocuments(view, currentIndex, currentPage);
}

function showBrowsePlaceholder() {
  document.getElementById("buckets-host").innerHTML = '<div class="empty-state">Select an index above to browse its buckets.</div>';
  document.getElementById("bucket-docs-host").innerHTML = "";
  document.getElementById("docs-host").innerHTML = '<div class="empty-state">Select an index above to browse its stored chunks.</div>';
  document.getElementById("pager-info").textContent = "";
}

async function deleteIndex(view, index) {
  const typed = prompt(
    `This permanently deletes ALL documents in "${index}". This cannot be undone.\n\nType the index name to confirm:`
  );
  if (typed !== index) {
    if (typed !== null) toast("Index name did not match — nothing deleted", "error");
    return;
  }
  try {
    await api(`/api/v1/indices/${encodeURIComponent(index)}`, { method: "DELETE" });
    toast(`Deleted index "${index}"`, "success");
    if (currentIndex === index) currentIndex = null;
    await refreshOverview(view);
  } catch (e) {
    toast(errorMessage(e), "error");
  }
}

// Build the "Create new index" sub-tab: name + chunking + embedding + files, then a fresh index
// is built on submit. Lazily invoked the first time the tab is opened. Behaviour matches the old
// Ingestion KB create flow.
async function buildCreateTab(view) {
  const host = document.getElementById("idx-create");

  let defaultStrategy = "sentence";
  let embeddingChoices = { watsonx: [], fastembed: [], litellm: [], "litellm-watsonx": [], "litellm-ollama": [] };
  let defaultEmbeddingProvider = "watsonx";
  let defaultModelByProvider = {};
  let vectorStore = "";
  let chunkDefaults = {};
  try {
    const cfg = await api("/api/v1/config");
    vectorStore = cfg.vector_store || "";
    defaultStrategy = cfg.default_chunking_strategy;
    embeddingChoices = cfg.embedding_model_choices || embeddingChoices;
    defaultEmbeddingProvider = cfg.embedding_provider || defaultEmbeddingProvider;
    // provider -> its configured model, served by EmbeddingFactory.default_models().
    defaultModelByProvider = cfg.embedding_default_models || defaultModelByProvider;
    chunkDefaults = {
      chunk_size: cfg.chunk_size, chunk_overlap: cfg.chunk_overlap,
      semantic_breakpoint_percentile: cfg.semantic_breakpoint_percentile,
      sentence_window_size: cfg.sentence_window_size,
      llm_chunk_max_size: cfg.llm_chunk_max_size, llm_chunk_window_size: cfg.llm_chunk_window_size,
    };
  } catch { /* config unavailable, fall back silently */ }
  let selectedStrategy = defaultStrategy;
  let selectedEmbeddingProvider = defaultEmbeddingProvider;

  let indices = [];
  try { indices = (await api("/api/v1/indices")).indices || []; } catch { /* none yet */ }
  const existingIndexNames = new Set(indices.map(ix => ix.index));

  const buckets = getKnownBuckets();
  const bucketChips = buckets.length ? `<div class="chip-row">${buckets.map(b => `<span class="pill" data-bucket-suggestion="${escapeHtml(b)}" style="cursor:pointer">${escapeHtml(b)}</span>`).join("")}</div>` : "";

  host.innerHTML = `
    <div class="chip-row" style="margin:0 0 1.5rem">
      <span class="pill pill-ok">🗄️ Active vector store: ${escapeHtml(vectorStore || "unknown")}</span>
    </div>

    <div class="card">
      <h2><span class="step-num">1</span> 📦 Index &amp; bucket</h2>
      <label class="field">
        <span class="label-text">Index name <span class="hint">(new index; suggested from strategy, editable)</span></span>
        <input type="text" id="create-index-name" placeholder="sentence">
      </label>
      <label class="field">
        <span class="label-text">Bucket / collection <span class="hint">(stored as metadata.bucket — filter on it later in Retrieval/Query)</span></span>
        <input type="text" id="create-bucket" placeholder="e.g. triage-protocols, hr-policies">
        ${bucketChips}
      </label>
    </div>

    <div class="card">
      <h2><span class="step-num">2</span> ✂️ Chunking</h2>
      <div class="strategy-picker" id="strategy-picker">
        ${STRATEGIES.map(s => strategyCardHtml(s, s.name === defaultStrategy)).join("")}
      </div>
      <div class="row" id="chunk-params" style="margin-top:.8rem"></div>
      <label class="field" style="margin-top:.8rem;display:flex;align-items:center;gap:.5rem;flex-direction:row">
        <input type="checkbox" id="enrich-keywords" style="width:auto">
        <span>LLM keyword enrichment <span class="hint">(slower — one LLM call per chunk; adds search keywords to boost recall)</span></span>
      </label>
    </div>

    <div class="card">
      <h2><span class="step-num">3</span> 🧮 Embedding model</h2>
      <label class="field">
        <span class="label-text">Provider</span>
        <select id="embedding-provider">
          ${Object.keys(embeddingChoices).map(p => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join("")}
        </select>
      </label>
      <label class="field">
        <span class="label-text">Model</span>
        <div id="embedding-model-host"></div>
      </label>
    </div>

    <div class="card">
      <h2><span class="step-num">4</span> 🏷️ Extra metadata <span class="hint">(optional, merged into every chunk alongside bucket)</span></h2>
      <div id="meta-rows"></div>
      <button type="button" class="secondary" id="add-meta-row">+ metadata field</button>
    </div>

    <div class="card">
      <h2>📄 Document(s)</h2>
      <label class="field">
        <div class="dropzone" id="create-dropzone">
          <div>Click to choose files, or drag them here</div>
          <div class="file-name" id="create-file-name-label"></div>
        </div>
        <input type="file" id="create-ingest-file" style="display:none" multiple>
      </label>
    </div>

    <div class="index-info-panel" id="create-info-panel"></div>

    <div class="btn-row" style="margin-bottom:.6rem">
      <button class="primary" id="create-submit">Create index &amp; ingest</button>
    </div>
    <div class="upload-progress" id="create-progress" style="display:none;margin-bottom:1.5rem">
      <div class="progress-bar-track"><div class="progress-bar-fill" id="create-progress-fill" style="width:0%"></div></div>
      <div class="progress-label"><span class="spinner"></span><span id="create-inline-status"></span></div>
    </div>

    <div class="btn-row" style="justify-content:space-between;align-items:center">
      <h2 style="font-size:1.05rem;margin:0">Jobs this session</h2>
      <button type="button" class="secondary" id="create-clear-jobs">Clear list</button>
    </div>
    <div id="create-job-history"></div>
  `;

  wireDropzone({
    input: document.getElementById("create-ingest-file"),
    dropzone: document.getElementById("create-dropzone"),
    label: document.getElementById("create-file-name-label"),
  });

  // --- Strategy picker + chunk params ---
  const picker = document.getElementById("strategy-picker");
  const chunkParamsHost = document.getElementById("chunk-params");
  const createIndexName = document.getElementById("create-index-name");
  const infoPanel = document.getElementById("create-info-panel");
  let indexNameEdited = false;
  createIndexName.oninput = () => { indexNameEdited = true; updateInfoPanel(); };
  function refreshChunkInputs() {
    chunkParamsHost.innerHTML = chunkInputsHtml(selectedStrategy, chunkDefaults);
    chunkParamsHost.querySelectorAll(".chunk-param").forEach(inp => inp.oninput = updateInfoPanel);
  }
  function refreshStrategySelection() {
    picker.querySelectorAll(".strategy-card").forEach(c => c.classList.toggle("selected", c.dataset.strategy === selectedStrategy));
  }
  // Keep the name in sync with the strategy until the user edits it by hand.
  function suggestIndexName() { if (!indexNameEdited) createIndexName.value = selectedStrategy; }
  picker.querySelectorAll(".strategy-card").forEach(c => {
    c.onclick = () => { selectedStrategy = c.dataset.strategy; refreshStrategySelection(); refreshChunkInputs(); suggestIndexName(); updateInfoPanel(); };
  });
  refreshStrategySelection();
  refreshChunkInputs();
  suggestIndexName();

  // --- Embedding provider + model control (curated list -> <select>, free-form -> text input) ---
  const providerSelect = document.getElementById("embedding-provider");
  const modelHost = document.getElementById("embedding-model-host");
  const embModelValue = () => document.getElementById("embedding-model").value.trim();
  function refreshModelOptions() {
    const provider = providerSelect.value;
    const models = embeddingChoices[provider] || [];
    const defaultModel = defaultModelByProvider[provider] || "";
    if (models.length) {
      modelHost.innerHTML = `<select id="embedding-model">${models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("")}</select>`;
      if (defaultModel && models.includes(defaultModel)) document.getElementById("embedding-model").value = defaultModel;
    } else {
      modelHost.innerHTML = `<input type="text" id="embedding-model" placeholder="any LiteLLM model, e.g. cohere/embed-multilingual-v3.0" value="${escapeHtml(defaultModel)}">
        <span class="hint">Free-form — any LiteLLM embedding model string; set its API key in Settings §3.</span>`;
    }
    const ctrl = document.getElementById("embedding-model");
    ctrl.oninput = updateInfoPanel;
    ctrl.onchange = updateInfoPanel;
  }
  providerSelect.value = selectedEmbeddingProvider;
  refreshModelOptions();
  providerSelect.onchange = () => { selectedEmbeddingProvider = providerSelect.value; refreshModelOptions(); updateInfoPanel(); };

  // --- Metadata rows + bucket suggestion chips ---
  const metaRows = document.getElementById("meta-rows");
  document.getElementById("add-meta-row").onclick = () => addMetaRow(metaRows);
  host.querySelectorAll("[data-bucket-suggestion]").forEach(chip => {
    chip.onclick = () => { document.getElementById("create-bucket").value = chip.dataset.bucketSuggestion; };
  });

  function readCreateChunkParams() {
    const out = {};
    chunkParamsHost.querySelectorAll(".chunk-param").forEach(inp => {
      if (inp.value !== "") out[inp.dataset.chunkKey] = Number(inp.value);
    });
    return out;
  }
  // Info panel previews the new index; a name clash disables submit.
  function updateInfoPanel() {
    const submitBtn = document.getElementById("create-submit");
    const name = createIndexName.value.trim() || selectedStrategy || "(unnamed)";
    const exists = existingIndexNames.has(name);
    submitBtn.disabled = exists;
    submitBtn.title = exists ? `"${name}" already exists — add to it from the Ingestion KB page, or rename` : "";
    const chunkBits = (CHUNK_FIELDS[selectedStrategy] || []).map(f => {
      const v = readCreateChunkParams()[f.key] ?? chunkDefaults[f.key];
      return `${f.key}=${v}`;
    });
    infoPanel.innerHTML = `
      <div class="info-title">🆕 New index</div>
      <div class="info-grid">
        <span><strong>index</strong> <code class="inline">${escapeHtml(name)}</code></span>
        <span><strong>vector store</strong> ${escapeHtml(vectorStore || "?")}</span>
        <span><strong>chunking</strong> ${escapeHtml([selectedStrategy, ...chunkBits].join(" · "))}</span>
        <span><strong>embedding</strong> ${escapeHtml(providerSelect.value)}/${escapeHtml(embModelValue())}</span>
      </div>
      ${exists ? `<div class="info-warn">⚠ An index named "${escapeHtml(name)}" already exists — add to it from the Ingestion KB page, or choose a different name.</div>` : ""}`;
  }

  // --- Job history + clear ---
  renderJobHistory(document.getElementById("create-job-history"));
  document.getElementById("create-clear-jobs").onclick = () => {
    clearJobHistory();
    renderJobHistory(document.getElementById("create-job-history"));
    toast("Job list cleared", "info");
  };
  updateInfoPanel();

  // --- Submit: build a new index ---
  document.getElementById("create-submit").onclick = async () => {
    const fileInput = document.getElementById("create-ingest-file");
    if (!fileInput.files.length) { toast("Choose a file first", "error"); return; }
    const indexName = createIndexName.value.trim() || selectedStrategy || "";
    if (!indexName) { toast("Enter an index name", "error"); return; }
    if (existingIndexNames.has(indexName)) { toast(`Index "${indexName}" already exists — add to it from the Ingestion KB page`, "error"); return; }
    const embModel = embModelValue();
    if (!embModel) { toast("Enter an embedding model", "error"); return; }

    const bucket = document.getElementById("create-bucket").value.trim();
    const metadata = {};
    if (bucket) metadata.bucket = bucket;
    metaRows.querySelectorAll(".filter-row").forEach(row => {
      const k = row.querySelector(".m-key").value.trim();
      const v = row.querySelector(".m-value").value;
      if (k) metadata[k] = v;
    });

    const progressBox = document.getElementById("create-progress");
    const progressFill = document.getElementById("create-progress-fill");
    const status = document.getElementById("create-inline-status");
    const submitBtn = document.getElementById("create-submit");
    const showProgress = (pct, label) => { progressBox.style.display = ""; progressFill.style.width = pct + "%"; status.textContent = label; };
    const hideProgress = () => { progressBox.style.display = "none"; progressFill.style.width = "0%"; status.textContent = ""; };

    submitBtn.disabled = true;
    await runIngestBatch(
      {
        files: fileInput.files,
        indexName,
        bucket,
        strategy: selectedStrategy,
        embProvider: providerSelect.value,
        embModel,
        chunkParams: readCreateChunkParams(),
        enrich: document.getElementById("enrich-keywords").checked,
        metadata,
      },
      {
        showProgress, hideProgress,
        onJobStarted: (job) => { addJobToHistory(job); renderJobHistory(document.getElementById("create-job-history")); },
        onBatchDone: () => { renderJobHistory(document.getElementById("create-job-history")); submitBtn.disabled = false; },
      }
    );
  };
}

export async function render(view) {
  currentIndex = null; currentPage = 0; currentBucket = null;
  view.innerHTML = `
    <h1 class="page-title">Indices</h1>
    <p class="page-sub">Create a new index from scratch, or explore what already exists across all vector stores.</p>

    <div class="mode-toggle" id="idx-mode-toggle">
      <button type="button" class="mode-btn" data-mode="create">✨ Create new index</button>
      <button type="button" class="mode-btn active" data-mode="explore">🔍 Explore existing</button>
      <button type="button" class="mode-btn" data-mode="visualize">📊 Visualizer</button>
    </div>

    <div id="idx-create" style="display:none"></div>
    <div id="idx-visualizer" style="display:none"></div>

    <div id="idx-explore">
      <div class="card kb-overview">
        <div class="btn-row" style="justify-content:space-between;align-items:center;margin-bottom:.6rem">
          <h2 style="margin:0">📚 Existing indexes <span class="hint" style="font-weight:400">— across all vector stores</span></h2>
          <button class="secondary" id="refresh-indices">Refresh</button>
        </div>
        <div id="overview-host"></div>
      </div>

      <div class="card">
        <h2 style="margin-top:0">Buckets</h2>
        <p class="page-sub" style="margin-top:0">Select a 📦 bucket to see the 📄 documents it contains.</p>
        <div id="buckets-host" class="bucket-card-grid"></div>
        <div id="bucket-docs-host"></div>
      </div>

      <div class="card">
        <h2 style="margin-top:0">All stored chunks</h2>
        <div class="btn-row" style="justify-content:space-between">
          <span id="pager-info" class="hint"></span>
          <div>
            <button class="secondary" id="prev-page">← Prev</button>
            <button class="secondary" id="next-page">Next →</button>
          </div>
        </div>
        <div id="docs-host" style="margin-top:.8rem"></div>
      </div>
    </div>
  `;

  // Sub-tab toggle: Explore is the default; the Create form is built lazily on first open.
  const createHost = document.getElementById("idx-create");
  const exploreHost = document.getElementById("idx-explore");
  const visualizerHost = document.getElementById("idx-visualizer");
  let createBuilt = false;
  let visualizerBuilt = false;
  document.querySelectorAll("#idx-mode-toggle .mode-btn").forEach(btn => {
    btn.onclick = () => {
      const m = btn.dataset.mode;
      document.querySelectorAll("#idx-mode-toggle .mode-btn").forEach(b => b.classList.toggle("active", b === btn));
      createHost.style.display = m === "create" ? "" : "none";
      exploreHost.style.display = m === "explore" ? "" : "none";
      visualizerHost.style.display = m === "visualize" ? "" : "none";
      if (m === "create" && !createBuilt) { buildCreateTab(view); createBuilt = true; }
      if (m === "visualize" && !visualizerBuilt) { buildVisualizerTab(view); visualizerBuilt = true; }
    };
  });

  document.getElementById("refresh-indices").onclick = () => refreshOverview(view);
  document.getElementById("prev-page").onclick = () => {
    if (currentIndex && currentPage > 0) { currentPage--; loadDocuments(view, currentIndex, currentPage); }
  };
  document.getElementById("next-page").onclick = () => {
    if (currentIndex) { currentPage++; loadDocuments(view, currentIndex, currentPage); }
  };

  try {
    await refreshOverview(view);
  } catch (e) {
    toast(errorMessage(e), "error");
  }
}
