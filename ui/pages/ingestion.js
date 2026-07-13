// Ingestion KB page: ADD documents to an EXISTING index. Config (chunking + embedding) is
// locked to whatever the picked index was built with; bucket (metadata.bucket) is per-document
// and can be new. Creating an index lives on the Indices page (Create-new sub-tab).
import { api, errorMessage, escapeHtml, toast, addJobToHistory, clearJobHistory, getKnownBuckets, indexOverviewHtml, wireIndicesOverview } from "../lib.js";
import { renderJobHistory, addMetaRow, wireDropzone, runIngestBatch, chunkingSummary } from "../ingest-common.js";

export async function render(view) {
  let vectorStore = "";
  try {
    const cfg = await api("/api/v1/config");
    vectorStore = cfg.vector_store || "";
  } catch { /* config unavailable, fall back silently */ }

  let indices = [];
  try { indices = (await api("/api/v1/indices")).indices || []; } catch { /* none yet */ }

  // Cross-store inventory for the "what already exists" panel (every vector store, not just active).
  let overview = null;
  try { overview = await api("/api/v1/indices/overview"); } catch { /* panel just won't render */ }

  const buckets = getKnownBuckets();
  const bucketChips = buckets.length ? `<div class="chip-row">${buckets.map(b => `<span class="pill" data-bucket-suggestion="${escapeHtml(b)}" style="cursor:pointer">${escapeHtml(b)}</span>`).join("")}</div>` : "";

  let lockedMeta = null; // the picked index's recorded config

  view.innerHTML = `
    <h1 class="page-title">Ingestion KB</h1>
    <p class="page-sub">Add documents to an existing index. Its chunking and embedding are locked to whatever built it — to create a new index, use the <a href="#/indices">Indices</a> page.</p>
    <div class="chip-row" style="margin:-1rem 0 1.5rem">
      <span class="pill pill-ok">🗄️ Active vector store: ${escapeHtml(vectorStore || "unknown")}</span>
    </div>

    <div class="card kb-overview">
      <h2>📚 Existing knowledge bases <span class="hint" style="font-weight:400">— across all vector stores</span></h2>
      <p class="hint" style="margin:-.3rem 0 1rem">What is already indexed. Click a bucket to list its documents.</p>
      ${indexOverviewHtml(overview)}
    </div>

    <div class="card">
      <h2><span class="step-num">1</span> 🗂️ Pick index</h2>
      <label class="field">
        <span class="label-text">Existing index <span class="hint">(its chunking + embedding lock to what built it)</span></span>
        <select id="add-index-select">
          <option value="">${indices.length ? "(choose an index)" : "no indices yet — create one on the Indices page"}</option>
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

  // --- File dropzone + existing-KB overview panel ---
  wireDropzone({
    input: document.getElementById("ingest-file"),
    dropzone: document.getElementById("dropzone"),
    label: document.getElementById("file-name-label"),
  });
  wireIndicesOverview(view, overview);

  const addIndexSelect = document.getElementById("add-index-select");
  const addLocked = document.getElementById("add-locked");
  const infoPanel = document.getElementById("index-info-panel");
  const addMetaRowsHost = document.getElementById("add-meta-rows");
  document.getElementById("add-add-meta-row").onclick = () => addMetaRow(addMetaRowsHost);

  // Info panel reflects the picked index's recorded config (nothing until one is chosen).
  function updateInfoPanel() {
    if (!lockedMeta) {
      infoPanel.innerHTML = `<div class="info-title">➕ Add to existing</div><div class="hint">Pick an index above to see its configuration.</div>`;
      return;
    }
    infoPanel.innerHTML = `
      <div class="info-title">➕ Adding to <code class="inline">${escapeHtml(lockedMeta.index)}</code></div>
      <div class="info-grid">
        <span><strong>vector store</strong> ${escapeHtml(vectorStore || "?")}</span>
        <span><strong>chunking</strong> ${escapeHtml(chunkingSummary(lockedMeta.chunking))}</span>
        <span><strong>embedding</strong> ${escapeHtml(lockedMeta.embedding_provider || "?")}/${escapeHtml(lockedMeta.embedding_model || "?")}</span>
        <span><strong>dim</strong> ${lockedMeta.dim ?? "?"}</span>
        <span><strong>docs</strong> ${lockedMeta.docs_count ?? "?"}</span>
        <span><strong>buckets</strong> ${(lockedMeta.buckets || []).map(escapeHtml).join(", ") || "—"}</span>
      </div>`;
  }

  // Picking an index fetches + locks its recorded build config.
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

  // Bucket suggestion chips.
  view.querySelectorAll("[data-bucket-suggestion]").forEach(chip => {
    chip.onclick = () => {
      document.getElementById("add-bucket").value = chip.dataset.bucketSuggestion;
      renderAddBuckets();
    };
  });

  // Renders the existing-buckets dropdown under the "Bucket & files" field: pick a bucket ->
  // it targets that bucket AND lists the docs already inside it, so the user sees what's there
  // and avoids re-uploading (dupes skip anyway). No-op until an index is picked.
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

  // --- Job history + clear ---
  renderJobHistory(document.getElementById("job-history"));
  document.getElementById("clear-jobs").onclick = () => {
    clearJobHistory();
    renderJobHistory(document.getElementById("job-history"));
    toast("Job list cleared", "info");
  };

  updateInfoPanel();

  // --- Submit: add to the locked index ---
  document.getElementById("submit-ingest").onclick = async () => {
    const fileInput = document.getElementById("ingest-file");
    if (!fileInput.files.length) { toast("Choose a file first", "error"); return; }
    if (!addIndexSelect.value || !lockedMeta) { toast("Pick an index to add to", "error"); return; }

    const bucket = document.getElementById("add-bucket").value.trim();
    // Send the recorded chunk knobs so the server's config-match guard passes (defaults may differ).
    const chunkParams = {};
    if (lockedMeta.chunking) {
      for (const [k, v] of Object.entries(lockedMeta.chunking)) if (k !== "strategy") chunkParams[k] = v;
    }
    const metadata = {};
    if (bucket) metadata.bucket = bucket;
    addMetaRowsHost.querySelectorAll(".filter-row").forEach(row => {
      const k = row.querySelector(".m-key").value.trim();
      const v = row.querySelector(".m-value").value;
      if (k) metadata[k] = v;
    });

    const progressBox = document.getElementById("upload-progress");
    const progressFill = document.getElementById("progress-bar-fill");
    const status = document.getElementById("ingest-inline-status");
    const submitBtn = document.getElementById("submit-ingest");
    const showProgress = (pct, label) => { progressBox.style.display = ""; progressFill.style.width = pct + "%"; status.textContent = label; };
    const hideProgress = () => { progressBox.style.display = "none"; progressFill.style.width = "0%"; status.textContent = ""; };

    submitBtn.disabled = true;
    await runIngestBatch(
      {
        files: fileInput.files,
        indexName: addIndexSelect.value,
        bucket,
        strategy: lockedMeta.chunking?.strategy || null,
        embProvider: lockedMeta.embedding_provider || null,
        embModel: lockedMeta.embedding_model || null,
        chunkParams,
        enrich: document.getElementById("add-enrich-keywords").checked,
        metadata,
      },
      {
        showProgress, hideProgress,
        onJobStarted: (job) => { addJobToHistory(job); renderJobHistory(document.getElementById("job-history")); },
        onBatchDone: () => { renderJobHistory(document.getElementById("job-history")); submitBtn.disabled = false; },
      }
    );
  };
}