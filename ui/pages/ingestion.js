// Ingestion KB page: two guided flows — CREATE a new index (free choice of name, chunking,
// chunk params, embedding model) or ADD documents to an existing KB (config locked to what the
// index was built with). An index = one chunking config + one embedding model; a persistent
// info panel always shows the resolved target index's configuration. Bucket (metadata.bucket)
// is a first-class field in both flows and scopes retrieval later.
import { api, errorMessage, escapeHtml, toast, getJobHistory, addJobToHistory, updateJobInHistory, clearJobHistory, getKnownBuckets, rememberBucket } from "../lib.js";

const STRATEGIES = [
  {
    name: "sentence",
    desc: "Splits on sentence boundaries. Simple, predictable, good default for prose.",
    example: "Il protocollo assegna codici colore.<mark>|</mark>In base alla gravita, i codici sono rosso e verde.<mark>|</mark>",
  },
  {
    name: "token",
    desc: "Fixed-size token windows with overlap. Good for dense technical text without clear sentence structure.",
    example: "[ tok1 tok2 tok3 tok4 ]<mark>|</mark>[ tok3 tok4 tok5 tok6 ]<mark>|</mark>  (overlap = 2)",
  },
  {
    name: "semantic",
    desc: "Groups sentences by embedding similarity; splits where meaning shifts, not at a fixed size.",
    example: "Il triage assegna codici colore. I codici indicano priorita.<mark>|</mark>La pizza napoletana usa farina e pomodoro.<mark>|</mark>",
  },
  {
    name: "window",
    desc: "One sentence per chunk, but stores surrounding sentences as context for the LLM at answer time.",
    example: "node: \"I codici indicano priorita.\"  window: [prev, <b>this</b>, next]",
  },
  {
    name: "llm",
    desc: "Uses the LLM to judge topic boundaries between sentences — produces the most semantically coherent chunks. Slower than other strategies (one LLM call per boundary).",
    example: "\"Il protocollo assegna codici colore. I codici indicano priorita.\"<mark>|</mark>\"La pizza napoletana usa farina…\"<mark>|</mark>",
    warn: "Significantly slower — one LLM call per sentence boundary.",
  },
];

// Which chunk knobs (Settings field names) are meaningful per strategy — mirrors the server's
// CHUNK_PARAM_FIELDS. Only these are shown/sent so an index's recorded config stays clean.
const CHUNK_FIELDS = {
  sentence: [
    { key: "chunk_size", label: "Chunk size (tokens)", min: 1, step: 1 },
    { key: "chunk_overlap", label: "Overlap (tokens)", min: 0, step: 1 },
  ],
  token: [
    { key: "chunk_size", label: "Chunk size (tokens)", min: 1, step: 1 },
    { key: "chunk_overlap", label: "Overlap (tokens)", min: 0, step: 1 },
  ],
  semantic: [
    { key: "semantic_breakpoint_percentile", label: "Breakpoint percentile", min: 1, max: 100, step: 1 },
  ],
  window: [
    { key: "sentence_window_size", label: "Window size (sentences)", min: 1, step: 1 },
  ],
  llm: [
    { key: "llm_chunk_max_size", label: "Max chunk size (tokens)", min: 1, step: 1 },
    { key: "llm_chunk_window_size", label: "Sentence window", min: 1, step: 1 },
  ],
};

let selectedStrategy = null;
let extraMetaRows = 0;
let pollTimers = {};

function strategyCardHtml(s, isDefault) {
  return `
    <div class="strategy-card" data-strategy="${s.name}">
      <div class="name">${s.name}${isDefault ? ' <span class="pill">default</span>' : ""}${s.warn ? ' <span class="pill pill-warn">slow</span>' : ""}</div>
      <div class="desc">${s.desc}</div>
      ${s.warn ? `<div class="strategy-warn">⚠ ${s.warn}</div>` : ""}
      <div class="example">${s.example}</div>
    </div>`;
}

function jobCardHtml(job) {
  const status = job.status || "pending";
  const running = status === "pending" || status === "running";
  // Live progress bar (reuses the upload bar CSS); completed/failed cards keep the plain layout
  const progressHtml = running ? `
      <div class="progress-bar-track" style="margin-top:.5rem"><div class="progress-bar-fill" style="width:${job.progress || 0}%"></div></div>
      <div class="hint">${escapeHtml(job.stage || "starting")} · ${job.progress || 0}%</div>` : "";
  return `
    <div class="job-card" data-job-id="${job.job_id}">
      <div class="top">
        <div><strong>${escapeHtml(job.file_name || "document")}</strong> <span class="hint">${escapeHtml(job.strategy || "")}${job.bucket ? " · bucket: " + escapeHtml(job.bucket) : ""}${job.embedding_model ? " · " + escapeHtml(job.embedding_model) : ""}</span></div>
        <span class="job-status ${status}">${status}</span>
      </div>
      <div class="job-meta" data-job-detail>${job.result ? (job.result.skipped_duplicate ? `duplicate skipped — same content already indexed as <strong>${escapeHtml(job.result.duplicate_of || "")}</strong> in <code class="inline">${job.result.index}</code>` : `${job.result.num_documents} doc → ${job.result.num_nodes} chunks in <code class="inline">${job.result.index}</code>`) : (job.detail ? escapeHtml(job.detail) : "job id: " + job.job_id)}</div>${progressHtml}
    </div>`;
}

async function pollJob(jobId, onUpdate) {
  clearInterval(pollTimers[jobId]);
  const tick = async () => {
    try {
      const res = await api(`/api/v1/ingest/${jobId}`);
      updateJobInHistory(jobId, res);
      onUpdate(res);
      if (res.status === "completed" || res.status === "failed") {
        clearInterval(pollTimers[jobId]);
        toast(res.status === "completed" ? "Ingestion completed" : "Ingestion failed", res.status === "completed" ? "success" : "error");
      }
    } catch (e) {
      clearInterval(pollTimers[jobId]);
    }
  };
  pollTimers[jobId] = setInterval(tick, 1500);
  tick();
}

function renderJobHistory(container) {
  const jobs = getJobHistory();
  if (!jobs.length) {
    container.innerHTML = '<div class="empty-state">No ingestion jobs yet this session.</div>';
    return;
  }
  container.innerHTML = jobs.map(jobCardHtml).join("");
  jobs.forEach(j => {
    if (j.status === "pending" || j.status === "running") {
      pollJob(j.job_id, (res) => {
        const card = container.querySelector(`[data-job-id="${j.job_id}"]`);
        if (card) card.outerHTML = jobCardHtml({ ...j, ...res });
      });
    }
  });
}

function addMetaRow(container) {
  const id = extraMetaRows++;
  const row = document.createElement("div");
  row.className = "filter-row";
  row.dataset.metaRow = id;
  row.innerHTML = `
    <input type="text" placeholder="key" class="m-key">
    <input type="text" placeholder="value" class="m-value">
    <button type="button" class="secondary" data-remove>×</button>`;
  row.querySelector("[data-remove]").onclick = () => row.remove();
  container.appendChild(row);
}

// Chunk-param number inputs for a strategy, prefilled from server defaults (or provided values).
function chunkInputsHtml(strategy, values) {
  const fields = CHUNK_FIELDS[strategy] || [];
  return fields.map(f => `
    <label class="field" style="margin:0">
      <span class="label-text">${f.label}</span>
      <input type="number" class="chunk-param" data-chunk-key="${f.key}" min="${f.min ?? 0}"${f.max ? ` max="${f.max}"` : ""} step="${f.step ?? 1}" value="${values[f.key] ?? ""}">
    </label>`).join("");
}

// One-line summary of a chunking config dict {strategy, ...knobs} for the info panel / locked view.
function chunkingSummary(chunking) {
  if (!chunking) return "unrecorded";
  const knobs = Object.entries(chunking).filter(([k]) => k !== "strategy").map(([k, v]) => `${k}=${v}`);
  return [chunking.strategy, ...knobs].join(" · ");
}

export async function render(view) {
  selectedStrategy = null;
  let defaultStrategy = "sentence";
  let embeddingChoices = { watsonx: [], fastembed: [] };
  let defaultEmbeddingProvider = "watsonx";
  let defaultModelByProvider = {};
  let indexPrefix = "";
  let chunkDefaults = {};
  try {
    const cfg = await api("/api/v1/config");
    defaultStrategy = cfg.default_chunking_strategy;
    embeddingChoices = cfg.embedding_model_choices || embeddingChoices;
    defaultEmbeddingProvider = cfg.embedding_provider || defaultEmbeddingProvider;
    defaultModelByProvider = { watsonx: cfg.watsonx_embedding_model_id, fastembed: cfg.fastembed_model_name };
    indexPrefix = cfg.opensearch_index_prefix || "";
    chunkDefaults = {
      chunk_size: cfg.chunk_size, chunk_overlap: cfg.chunk_overlap,
      semantic_breakpoint_percentile: cfg.semantic_breakpoint_percentile,
      sentence_window_size: cfg.sentence_window_size,
      llm_chunk_max_size: cfg.llm_chunk_max_size, llm_chunk_window_size: cfg.llm_chunk_window_size,
    };
  } catch { /* config unavailable, fall back silently */ }
  selectedStrategy = defaultStrategy;
  let selectedEmbeddingProvider = defaultEmbeddingProvider;

  let indices = [];
  try { indices = (await api("/api/v1/indices")).indices || []; } catch { /* none yet */ }

  const buckets = getKnownBuckets();
  const bucketChips = buckets.length ? `<div class="chip-row">${buckets.map(b => `<span class="pill" data-bucket-suggestion="${escapeHtml(b)}" style="cursor:pointer">${escapeHtml(b)}</span>`).join("")}</div>` : "";

  let mode = "create"; // "create" | "add"
  let lockedMeta = null; // in add-mode: the picked index's recorded config

  view.innerHTML = `
    <h1 class="page-title">Ingestion KB</h1>
    <p class="page-sub">Two flows: build a brand-new index, or add documents to an existing one. An index is locked to a single chunking config and embedding model — to change either, create a new index.</p>

    <div class="mode-toggle" id="mode-toggle">
      <button type="button" class="mode-btn active" data-mode="create">✨ Create new index</button>
      <button type="button" class="mode-btn" data-mode="add">➕ Add to existing KB</button>
    </div>

    <!-- CREATE MODE -->
    <div id="mode-create">
      <div class="card">
        <h2><span class="step-num">1</span> 📦 Index &amp; bucket</h2>
        <label class="field">
          <span class="label-text">Index name <span class="hint">(new index; suggested from prefix + strategy, editable)</span></span>
          <input type="text" id="create-index-name" placeholder="${escapeHtml(indexPrefix)}_sentence">
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
            <option value="watsonx">watsonx</option>
            <option value="fastembed">fastembed</option>
          </select>
        </label>
        <label class="field">
          <span class="label-text">Model</span>
          <select id="embedding-model"></select>
        </label>
      </div>

      <div class="card">
        <h2><span class="step-num">4</span> 🏷️ Extra metadata <span class="hint">(optional, merged into every chunk alongside bucket)</span></h2>
        <div id="meta-rows"></div>
        <button type="button" class="secondary" id="add-meta-row">+ metadata field</button>
      </div>
    </div>

    <!-- ADD MODE -->
    <div id="mode-add" style="display:none">
      <div class="card">
        <h2><span class="step-num">1</span> 🗂️ Pick index</h2>
        <label class="field">
          <span class="label-text">Existing index <span class="hint">(its chunking + embedding lock to what built it)</span></span>
          <select id="add-index-select">
            <option value="">${indices.length ? "(choose an index)" : "no indices yet — create one first"}</option>
            ${indices.map(ix => `<option value="${escapeHtml(ix.index)}">${escapeHtml(ix.index)} (${ix["docs.count"]} docs)</option>`).join("")}
          </select>
        </label>
        <div id="add-locked" class="locked-config" style="display:none"></div>
      </div>

      <div class="card">
        <h2><span class="step-num">2</span> 📦 Bucket &amp; files</h2>
        <label class="field">
          <span class="label-text">Bucket / collection <span class="hint">(per-document — you can add a new bucket to an existing index)</span></span>
          <input type="text" id="add-bucket" placeholder="e.g. triage-protocols, hr-policies">
          ${bucketChips}
        </label>
        <div id="add-bucket-existing"></div>
      </div>

      <div class="card">
        <h2><span class="step-num">3</span> 🏷️ Extra metadata &amp; enrichment</h2>
        <div id="add-meta-rows"></div>
        <button type="button" class="secondary" id="add-add-meta-row">+ metadata field</button>
        <label class="field" style="margin-top:.8rem;display:flex;align-items:center;gap:.5rem;flex-direction:row">
          <input type="checkbox" id="add-enrich-keywords" style="width:auto">
          <span>LLM keyword enrichment <span class="hint">(slower — adds search keywords per chunk)</span></span>
        </label>
      </div>
    </div>

    <!-- SHARED: file, info panel, submit, jobs -->
    <div class="card">
      <h2>📄 Document(s)</h2>
      <label class="field">
        <div class="dropzone" id="dropzone">
          <div>Click to choose files, or drag them here</div>
          <div class="file-name" id="file-name-label"></div>
        </div>
        <input type="file" id="ingest-file" style="display:none" multiple>
      </label>
    </div>

    <div class="index-info-panel" id="index-info-panel"></div>

    <div class="btn-row" style="margin-bottom:.6rem">
      <button class="primary" id="submit-ingest">Upload &amp; ingest</button>
    </div>
    <div class="upload-progress" id="upload-progress" style="display:none;margin-bottom:1.5rem">
      <div class="progress-bar-track"><div class="progress-bar-fill" id="progress-bar-fill" style="width:0%"></div></div>
      <div class="progress-label"><span class="spinner" id="upload-spinner"></span><span id="ingest-inline-status"></span></div>
    </div>

    <div class="btn-row" style="justify-content:space-between;align-items:center">
      <h2 style="font-size:1.05rem;margin:0">Jobs this session</h2>
      <button type="button" class="secondary" id="clear-jobs">Clear list</button>
    </div>
    <div id="job-history"></div>
  `;

  // --- Shared file dropzone ---
  const fileInput = document.getElementById("ingest-file");
  const dropzone = document.getElementById("dropzone");
  const fileLabel = document.getElementById("file-name-label");
  const fileLabelText = (files) =>
    files.length > 1 ? `${files.length} files: ${[...files].map(f => f.name).join(", ")}` : (files[0]?.name || "");
  dropzone.onclick = () => fileInput.click();
  fileInput.onchange = () => { fileLabel.textContent = fileLabelText(fileInput.files); };
  ["dragover", "dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.toggle("drag", evt === "dragover");
      if (evt === "drop" && e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileLabel.textContent = fileLabelText(e.dataTransfer.files);
      }
    });
  });

  // --- CREATE: strategy picker + chunk params ---
  const picker = document.getElementById("strategy-picker");
  const chunkParamsHost = document.getElementById("chunk-params");
  const createIndexName = document.getElementById("create-index-name");
  let indexNameEdited = false;
  createIndexName.oninput = () => { indexNameEdited = true; updateInfoPanel(); };
  function refreshChunkInputs() {
    chunkParamsHost.innerHTML = chunkInputsHtml(selectedStrategy, chunkDefaults);
    chunkParamsHost.querySelectorAll(".chunk-param").forEach(inp => inp.oninput = updateInfoPanel);
  }
  function refreshStrategySelection() {
    picker.querySelectorAll(".strategy-card").forEach(c => c.classList.toggle("selected", c.dataset.strategy === selectedStrategy));
  }
  function suggestIndexName() {
    // Keep the name in sync with the strategy until the user edits it by hand
    if (!indexNameEdited && indexPrefix) createIndexName.value = `${indexPrefix}_${selectedStrategy}`;
  }
  picker.querySelectorAll(".strategy-card").forEach(c => {
    c.onclick = () => { selectedStrategy = c.dataset.strategy; refreshStrategySelection(); refreshChunkInputs(); suggestIndexName(); updateInfoPanel(); };
  });
  refreshStrategySelection();
  refreshChunkInputs();
  suggestIndexName();

  // --- CREATE: embedding selects ---
  const providerSelect = document.getElementById("embedding-provider");
  const modelSelect = document.getElementById("embedding-model");
  function refreshModelOptions() {
    const provider = providerSelect.value;
    const models = embeddingChoices[provider] || [];
    modelSelect.innerHTML = models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
    const defaultModel = defaultModelByProvider[provider];
    if (defaultModel && models.includes(defaultModel)) modelSelect.value = defaultModel;
  }
  providerSelect.value = selectedEmbeddingProvider;
  refreshModelOptions();
  providerSelect.onchange = () => { selectedEmbeddingProvider = providerSelect.value; refreshModelOptions(); updateInfoPanel(); };
  modelSelect.onchange = updateInfoPanel;

  // --- ADD: index picker (fetch + lock its recorded config) ---
  const addIndexSelect = document.getElementById("add-index-select");
  const addLocked = document.getElementById("add-locked");
  addIndexSelect.onchange = async () => {
    lockedMeta = null;
    addLocked.style.display = "none";
    if (!addIndexSelect.value) { updateInfoPanel(); return; }
    try {
      lockedMeta = await api(`/api/v1/indices/${encodeURIComponent(addIndexSelect.value)}/info`);
      addLocked.style.display = "";
      addLocked.innerHTML = lockedMeta.chunking
        ? `🔒 Locked to this index's build config:<br><strong>chunking</strong> ${escapeHtml(chunkingSummary(lockedMeta.chunking))}<br><strong>embedding</strong> ${escapeHtml(lockedMeta.embedding_provider || "?")}/${escapeHtml(lockedMeta.embedding_model || "?")}`
        : `⚠ Legacy index with no recorded build config — documents will be added with the server's current defaults.`;
    } catch (e) {
      toast(errorMessage(e), "error");
    }
    renderAddBuckets();
    updateInfoPanel();
  };
  // typing a bucket re-syncs the existing-bucket dropdown + its doc list.
  document.getElementById("add-bucket").oninput = renderAddBuckets;

  // --- Bucket suggestion chips (both modes) ---
  view.querySelectorAll("[data-bucket-suggestion]").forEach(chip => {
    chip.onclick = () => {
      const target = mode === "create" ? "create-bucket" : "add-bucket";
      document.getElementById(target).value = chip.dataset.bucketSuggestion;
      if (mode === "add") renderAddBuckets();
    };
  });

  // --- Metadata rows (separate hosts per mode) ---
  const metaRows = document.getElementById("meta-rows");
  document.getElementById("add-meta-row").onclick = () => addMetaRow(metaRows);
  const addMetaRows = document.getElementById("add-meta-rows");
  document.getElementById("add-add-meta-row").onclick = () => addMetaRow(addMetaRows);

  // --- Info panel: reflects the resolved target index for the active mode ---
  const infoPanel = document.getElementById("index-info-panel");
  const existingIndexNames = new Set(indices.map(ix => ix.index));
  function readCreateChunkParams() {
    const out = {};
    chunkParamsHost.querySelectorAll(".chunk-param").forEach(inp => {
      if (inp.value !== "") out[inp.dataset.chunkKey] = Number(inp.value);
    });
    return out;
  }
  function updateInfoPanel() {
    if (mode === "create") {
      const name = createIndexName.value.trim() || (indexPrefix ? `${indexPrefix}_${selectedStrategy}` : "(unnamed)");
      const exists = existingIndexNames.has(name);
      const chunkBits = (CHUNK_FIELDS[selectedStrategy] || []).map(f => {
        const v = readCreateChunkParams()[f.key] ?? chunkDefaults[f.key];
        return `${f.key}=${v}`;
      });
      infoPanel.innerHTML = `
        <div class="info-title">🆕 New index</div>
        <div class="info-grid">
          <span><strong>index</strong> <code class="inline">${escapeHtml(name)}</code></span>
          <span><strong>chunking</strong> ${escapeHtml([selectedStrategy, ...chunkBits].join(" · "))}</span>
          <span><strong>embedding</strong> ${escapeHtml(providerSelect.value)}/${escapeHtml(modelSelect.value)}</span>
        </div>
        ${exists ? `<div class="info-warn">⚠ An index named "${escapeHtml(name)}" already exists — use “Add to existing KB” to add documents, or choose a different name.</div>` : ""}`;
    } else {
      if (!lockedMeta) {
        infoPanel.innerHTML = `<div class="info-title">➕ Add to existing</div><div class="hint">Pick an index above to see its configuration.</div>`;
        return;
      }
      infoPanel.innerHTML = `
        <div class="info-title">➕ Adding to <code class="inline">${escapeHtml(lockedMeta.index)}</code></div>
        <div class="info-grid">
          <span><strong>chunking</strong> ${escapeHtml(chunkingSummary(lockedMeta.chunking))}</span>
          <span><strong>embedding</strong> ${escapeHtml(lockedMeta.embedding_provider || "?")}/${escapeHtml(lockedMeta.embedding_model || "?")}</span>
          <span><strong>dim</strong> ${lockedMeta.dim ?? "?"}</span>
          <span><strong>docs</strong> ${lockedMeta.docs_count ?? "?"}</span>
          <span><strong>buckets</strong> ${(lockedMeta.buckets || []).map(escapeHtml).join(", ") || "—"}</span>
        </div>`;
    }
  }

  // Renders the existing-buckets dropdown under the "Bucket & files" field (add-to-existing flow):
  // pick a bucket -> it targets that bucket AND lists the docs already inside it, so the user sees
  // what's there and avoids re-uploading (dupes skip anyway). No-op until an index is picked.
  function renderAddBuckets() {
    const host = document.getElementById("add-bucket-existing");
    if (!host) return;
    const bucketFiles = (lockedMeta && lockedMeta.bucket_files) || {};
    const names = Object.keys(bucketFiles);
    if (!lockedMeta || !names.length) {
      host.innerHTML = lockedMeta ? `<div class="hint" style="margin-top:6px">No documents yet — this index is empty.</div>` : "";
      return;
    }
    const selectedBucket = document.getElementById("add-bucket").value.trim();
    const options = names.map(b =>
      `<option value="${escapeHtml(b)}"${b === selectedBucket ? " selected" : ""}>📦 ${escapeHtml(b)} (${(bucketFiles[b] || []).length})</option>`
    ).join("");
    const files = bucketFiles[selectedBucket] || [];
    const docsList = !names.includes(selectedBucket) ? ""
      : files.length ? files.map(f => `<li>📄 ${escapeHtml(f)}</li>`).join("")
      : `<li class="hint">empty</li>`;
    host.innerHTML = `
      <div class="bucket-files-picker">
        <label class="bucket-cards-label hint">Or pick an existing bucket <span class="hint">— shows its docs; already-listed files skip as duplicates</span>
          <select class="bucket-files-select">
            <option value="">(choose a bucket)</option>${options}
          </select>
        </label>
        <ul class="bucket-card-files bucket-files-docs">${docsList}</ul>
      </div>`;
    host.querySelector(".bucket-files-select").onchange = e => {
      document.getElementById("add-bucket").value = e.target.value;
      renderAddBuckets();
    };
  }

  // --- Mode toggle ---
  const modeCreate = document.getElementById("mode-create");
  const modeAdd = document.getElementById("mode-add");
  document.querySelectorAll("#mode-toggle .mode-btn").forEach(btn => {
    btn.onclick = () => {
      mode = btn.dataset.mode;
      document.querySelectorAll("#mode-toggle .mode-btn").forEach(b => b.classList.toggle("active", b === btn));
      modeCreate.style.display = mode === "create" ? "" : "none";
      modeAdd.style.display = mode === "add" ? "" : "none";
      updateInfoPanel();
    };
  });

  // --- Job history + clear ---
  renderJobHistory(document.getElementById("job-history"));
  document.getElementById("clear-jobs").onclick = () => {
    Object.values(pollTimers).forEach(clearInterval);
    pollTimers = {};
    clearJobHistory();
    renderJobHistory(document.getElementById("job-history"));
    toast("Job list cleared", "info");
  };

  updateInfoPanel();

  // --- Submit ---
  document.getElementById("submit-ingest").onclick = async () => {
    if (!fileInput.files.length) { toast("Choose a file first", "error"); return; }

    // Resolve the request shape from the active mode
    let indexName, bucket, strategy, embProvider, embModel, chunkParams, enrich, metaHost;
    if (mode === "create") {
      indexName = createIndexName.value.trim() || (indexPrefix ? `${indexPrefix}_${selectedStrategy}` : "");
      if (!indexName) { toast("Enter an index name", "error"); return; }
      if (existingIndexNames.has(indexName)) { toast(`Index "${indexName}" already exists — use “Add to existing KB”`, "error"); return; }
      bucket = document.getElementById("create-bucket").value.trim();
      strategy = selectedStrategy;
      embProvider = providerSelect.value;
      embModel = modelSelect.value;
      chunkParams = readCreateChunkParams();
      enrich = document.getElementById("enrich-keywords").checked;
      metaHost = metaRows;
    } else {
      if (!addIndexSelect.value || !lockedMeta) { toast("Pick an index to add to", "error"); return; }
      indexName = addIndexSelect.value;
      bucket = document.getElementById("add-bucket").value.trim();
      strategy = lockedMeta.chunking?.strategy || null;
      embProvider = lockedMeta.embedding_provider || null;
      embModel = lockedMeta.embedding_model || null;
      // Send the recorded chunk knobs so the server's config-match guard passes (defaults may differ)
      chunkParams = {};
      if (lockedMeta.chunking) {
        for (const [k, v] of Object.entries(lockedMeta.chunking)) if (k !== "strategy") chunkParams[k] = v;
      }
      enrich = document.getElementById("add-enrich-keywords").checked;
      metaHost = addMetaRows;
    }

    const metadata = {};
    if (bucket) metadata.bucket = bucket;
    metaHost.querySelectorAll(".filter-row").forEach(row => {
      const k = row.querySelector(".m-key").value.trim();
      const v = row.querySelector(".m-value").value;
      if (k) metadata[k] = v;
    });

    const progressBox = document.getElementById("upload-progress");
    const progressFill = document.getElementById("progress-bar-fill");
    const status = document.getElementById("ingest-inline-status");
    const submitBtn = document.getElementById("submit-ingest");

    function showProgress(pct, label) { progressBox.style.display = ""; progressFill.style.width = pct + "%"; status.textContent = label; }
    function hideProgress() { progressBox.style.display = "none"; progressFill.style.width = "0%"; status.textContent = ""; }

    // One POST per file; index/strategy/embedding/chunk/metadata/enrichment shared across the batch
    const uploadOne = (file, label) => new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      if (strategy) form.append("strategy", strategy);
      if (indexName) form.append("index_name", indexName);
      if (embProvider) form.append("embedding_provider", embProvider);
      if (embModel) form.append("embedding_model", embModel);
      if (Object.keys(chunkParams).length) form.append("chunk_params", JSON.stringify(chunkParams));
      if (Object.keys(metadata).length) form.append("metadata", JSON.stringify(metadata));
      if (enrich) form.append("enrich_keywords", "true");
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/v1/ingest");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) { const pct = Math.round((e.loaded / e.total) * 100); showProgress(pct, `${label} ${pct}%`); }
      };
      xhr.onload = () => {
        let body;
        try { body = JSON.parse(xhr.responseText); } catch { body = xhr.responseText; }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else { const err = new Error(`HTTP ${xhr.status}`); err.body = body; reject(err); }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(form);
    });

    // Resolve once the job reaches a terminal state — used to serialize the batch
    const jobDone = (jobId) => new Promise((resolve) => {
      const timer = setInterval(async () => {
        try {
          const res = await api(`/api/v1/ingest/${jobId}`);
          if (res.status === "completed" || res.status === "failed") { clearInterval(timer); resolve(res); }
        } catch { clearInterval(timer); resolve(null); }
      }, 1500);
    });

    submitBtn.disabled = true;
    const files = [...fileInput.files];
    let started = 0;
    for (let i = 0; i < files.length; i++) {
      const tag = files.length > 1 ? `${i + 1}/${files.length} — ${files[i].name}` : "";
      showProgress(0, `uploading ${tag}…`);
      try {
        const res = await uploadOne(files[i], `uploading ${tag}`);
        started++;
        addJobToHistory({
          job_id: res.job_id, status: res.status,
          file_name: files[i].name, strategy, bucket,
          embedding_model: `${embProvider}/${embModel}`,
        });
        renderJobHistory(document.getElementById("job-history"));
        // Strictly one file at a time: wait for this ingestion before the next upload
        showProgress(100, `ingesting ${tag || files[i].name}…`);
        await jobDone(res.job_id);
      } catch (e) {
        // One bad file must not stop the batch — report and continue
        toast(`${files[i].name}: ${errorMessage(e)}`, "error");
      }
    }
    if (bucket && started) rememberBucket(bucket);
    if (started) toast(files.length > 1 ? `${started}/${files.length} ingestion jobs finished` : "Ingestion finished", "success");
    hideProgress();
    renderJobHistory(document.getElementById("job-history"));
    submitBtn.disabled = false;
  };
}
