// Shared ingest UI: strategy/chunk widgets, session job history, dropzone wiring, and the
// upload-and-poll batch loop. Used by both the Ingestion KB (add-to-existing) page and the
// Indices → Create-new sub-tab so the two flows stay in lockstep without duplicating code.
import { api, errorMessage, escapeHtml, toast, getJobHistory, updateJobInHistory, rememberBucket } from "./lib.js";

// Chunking strategies shown as selectable cards in the create flow, each with a short
// description and a tiny illustrative split example (mark = chunk boundary).
export const STRATEGIES = [
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
export const CHUNK_FIELDS = {
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

// Module-level metadata-row counter + poll timers, shared by the helpers below.
let extraMetaRows = 0;
let pollTimers = {};

// Markup for one selectable strategy card, flagged as default/slow where relevant.
export function strategyCardHtml(s, isDefault) {
  return `
    <div class="strategy-card" data-strategy="${s.name}">
      <div class="name">${s.name}${isDefault ? ' <span class="pill">default</span>' : ""}${s.warn ? ' <span class="pill pill-warn">slow</span>' : ""}</div>
      <div class="desc">${s.desc}</div>
      ${s.warn ? `<div class="strategy-warn">⚠ ${s.warn}</div>` : ""}
      <div class="example">${s.example}</div>
    </div>`;
}

// One session job-history card; live cards get a progress bar, terminal ones a result summary.
export function jobCardHtml(job) {
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

// Poll a single ingest job until it reaches a terminal state, invoking onUpdate each tick.
export async function pollJob(jobId, onUpdate) {
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

// Render the session's job history into a container, re-polling any still-running jobs.
export function renderJobHistory(container) {
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

// Append a removable key/value metadata input row to a container.
export function addMetaRow(container) {
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
export function chunkInputsHtml(strategy, values) {
  const fields = CHUNK_FIELDS[strategy] || [];
  return fields.map(f => `
    <label class="field" style="margin:0">
      <span class="label-text">${f.label}</span>
      <input type="number" class="chunk-param" data-chunk-key="${f.key}" min="${f.min ?? 0}"${f.max ? ` max="${f.max}"` : ""} step="${f.step ?? 1}" value="${values[f.key] ?? ""}">
    </label>`).join("");
}

// One-line summary of a chunking config dict {strategy, ...knobs} for the info panel / locked view.
export function chunkingSummary(chunking) {
  if (!chunking) return "unrecorded";
  const knobs = Object.entries(chunking).filter(([k]) => k !== "strategy").map(([k, v]) => `${k}=${v}`);
  return [chunking.strategy, ...knobs].join(" · ");
}

// Wire a file dropzone: click-to-choose + drag/drop, updating a label with the selected
// file name(s). Shared by both ingest flows.
export function wireDropzone({ input, dropzone, label }) {
  const fileLabelText = (files) =>
    files.length > 1 ? `${files.length} files: ${[...files].map(f => f.name).join(", ")}` : (files[0]?.name || "");
  dropzone.onclick = () => input.click();
  input.onchange = () => { label.textContent = fileLabelText(input.files); };
  ["dragover", "dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.toggle("drag", evt === "dragover");
      if (evt === "drop" && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        label.textContent = fileLabelText(e.dataTransfer.files);
      }
    });
  });
}

// Upload + ingest a batch of files against one resolved target config. One POST per file,
// strictly serial (each job polled to a terminal state before the next upload). A failing
// file toasts and the batch continues. The caller owns the progress DOM via `hooks`.
export async function runIngestBatch(config, hooks) {
  const { files, indexName, bucket, strategy, embProvider, embModel, chunkParams, enrich, metadata } = config;
  const { showProgress, hideProgress, onJobStarted, onBatchDone } = hooks;

  // One POST per file; index/strategy/embedding/chunk/metadata/enrichment shared across the batch.
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

  // Resolve once a job reaches a terminal state — used to serialize the batch.
  const jobDone = (jobId) => new Promise((resolve) => {
    const timer = setInterval(async () => {
      try {
        const res = await api(`/api/v1/ingest/${jobId}`);
        if (res.status === "completed" || res.status === "failed") { clearInterval(timer); resolve(res); }
      } catch { clearInterval(timer); resolve(null); }
    }, 1500);
  });

  const fileArr = [...files];
  let started = 0;
  for (let i = 0; i < fileArr.length; i++) {
    const tag = fileArr.length > 1 ? `${i + 1}/${fileArr.length} — ${fileArr[i].name}` : "";
    showProgress(0, `uploading ${tag}…`);
    try {
      const res = await uploadOne(fileArr[i], `uploading ${tag}`);
      started++;
      onJobStarted({
        job_id: res.job_id, status: res.status,
        file_name: fileArr[i].name, strategy, bucket,
        embedding_model: `${embProvider}/${embModel}`,
      });
      // Strictly one file at a time: wait for this ingestion before the next upload.
      showProgress(100, `ingesting ${tag || fileArr[i].name}…`);
      await jobDone(res.job_id);
    } catch (e) {
      // One bad file must not stop the batch — report and continue.
      toast(`${fileArr[i].name}: ${errorMessage(e)}`, "error");
    }
  }
  if (bucket && started) rememberBucket(bucket);
  if (started) toast(fileArr.length > 1 ? `${started}/${fileArr.length} ingestion jobs finished` : "Ingestion finished", "success");
  hideProgress();
  onBatchDone(started);
}