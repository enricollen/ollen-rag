// Shared helpers: same-origin fetch wrapper, toasts, localStorage-backed
// session state (job history, seen buckets), and small DOM/format utilities.

// Fetches a JSON (or form) API endpoint and throws with the parsed error
// body attached, so callers can render {error_code, detail} from the backend.
export async function api(path, options = {}) {
  const res = await fetch(path, options);
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}`);
    err.body = body;
    throw err;
  }
  return body;
}

// Renders whatever shape a failed api() call produced into a short string.
export function errorMessage(err) {
  if (err?.body?.detail) return err.body.detail;
  if (typeof err?.body === "string" && err.body) return err.body;
  return err?.message || "unexpected error";
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// --- Toasts -----------------------------------------------------------
export function toast(message, kind = "info") {
  const host = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// --- Session-local persisted state (survives page nav, not server-backed) --
const STORE_KEY = "ollen_rag_ui_state_v1";

function loadStore() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
}
function saveStore(state) {
  localStorage.setItem(STORE_KEY, JSON.stringify(state));
}

export function getJobHistory() {
  return loadStore().jobs || [];
}

export function addJobToHistory(job) {
  const state = loadStore();
  state.jobs = [job, ...(state.jobs || [])].slice(0, 30);
  saveStore(state);
}

export function updateJobInHistory(jobId, patch) {
  const state = loadStore();
  state.jobs = (state.jobs || []).map(j => (j.job_id === jobId ? { ...j, ...patch } : j));
  saveStore(state);
}

// Wipes the local ingestion job history (UI-only; does not touch indexed data).
export function clearJobHistory() {
  const state = loadStore();
  state.jobs = [];
  saveStore(state);
}

// --- Preloaded index/bucket lists for dropdown pickers ------------------
export async function fetchIndexList() {
  return (await api("/api/v1/indices")).indices;
}

// Build the <option> list for an index <select>, pre-selecting the index with the MOST
// documents. This prevents an empty/stray index (which sorts first in _cat output) from
// silently hijacking the default selection and making retrieval return 0 chunks.
export function indexOptionsHtml(indices) {
  if (!indices || !indices.length) return '<option value="">no indices yet</option>';
  const top = indices.reduce((a, b) => (Number(b["docs.count"]) > Number(a["docs.count"]) ? b : a));
  return indices.map(ix =>
    `<option value="${escapeHtml(ix.index)}"${ix.index === top.index ? " selected" : ""}>${escapeHtml(ix.index)} (${ix["docs.count"]} docs)</option>`
  ).join("");
}

export async function fetchBucketList(indexName) {
  return (await api(`/api/v1/indices/${encodeURIComponent(indexName)}/buckets`)).buckets;
}

export function getKnownBuckets() {
  return loadStore().buckets || [];
}

export function rememberBucket(bucket) {
  if (!bucket) return;
  const state = loadStore();
  const buckets = new Set(state.buckets || []);
  buckets.add(bucket);
  state.buckets = [...buckets].slice(-20);
  saveStore(state);
}

// --- Index info panel (shared by Query/Eval so both are explicit about the index's locks) ---

// Full recorded build config of an index (embedding + chunking + dim + docs + buckets).
export async function fetchIndexInfo(indexName) {
  return api(`/api/v1/indices/${encodeURIComponent(indexName)}/info`);
}

// One-line summary of a chunking config dict {strategy, ...knobs}.
function chunkingSummary(chunking) {
  if (!chunking) return "unrecorded (legacy index)";
  const knobs = Object.entries(chunking).filter(([k]) => k !== "strategy").map(([k, v]) => `${k}=${v}`);
  return [chunking.strategy, ...knobs].join(" · ");
}

// Renders the shared index-info panel HTML. `verb` names what runs against it (e.g. "Queries",
// "Eval cases") so the lock note reads naturally. Makes the embedding-model lock explicit: the
// pipeline always uses the model the index was built with, never a mixed one.
export function indexInfoHtml(info, verb = "Queries", showBucketFiles = true) {
  if (!info) return "";
  const emb = info.embedding_provider ? `${info.embedding_provider}/${info.embedding_model}` : "unrecorded (legacy index)";
  return `
    <div class="info-title">🔒 <code class="inline">${escapeHtml(info.index)}</code> — locked configuration</div>
    <div class="info-grid">
      <span><strong>chunking</strong> ${escapeHtml(chunkingSummary(info.chunking))}</span>
      <span><strong>embedding</strong> ${escapeHtml(emb)}</span>
      <span><strong>dim</strong> ${info.dim ?? "?"}</span>
      <span><strong>docs</strong> ${info.docs_count ?? "?"}</span>
      <span><strong>buckets</strong> ${(info.buckets || []).map(escapeHtml).join(", ") || "—"}</span>
    </div>
    ${showBucketFiles ? bucketFilesMapHtml(info.bucket_files || {}) : ""}
    <div class="info-note">${escapeHtml(verb)} run against this index use its recorded embedding model — no model mixing.</div>`;
}

// Read-only 📦 bucket -> 📄 documents picker: a dropdown of buckets; picking one lists its
// documents right underneath. Data is stashed on the <select> so wireBucketFiles() can render
// the doc list without a refetch. Call wireBucketFiles(container) after inserting the HTML.
function bucketFilesMapHtml(bucketFiles) {
  const names = Object.keys(bucketFiles);
  if (!names.length) return "";
  const options = names.map(b => `<option value="${escapeHtml(b)}">📦 ${escapeHtml(b)} (${(bucketFiles[b] || []).length})</option>`).join("");
  return `
    <div class="bucket-files-picker">
      <label class="bucket-cards-label hint">Buckets &amp; documents in this index
        <select class="bucket-files-select" data-bucket-files="${escapeHtml(JSON.stringify(bucketFiles))}">
          <option value="">(choose a bucket)</option>${options}
        </select>
      </label>
      <ul class="bucket-card-files bucket-files-docs"></ul>
    </div>`;
}

// Wires every bucket dropdown inside `root`: on change, renders the selected bucket's document
// list into the sibling .bucket-files-docs. Idempotent — safe to call after each innerHTML set.
export function wireBucketFiles(root) {
  root.querySelectorAll(".bucket-files-select").forEach(sel => {
    const map = JSON.parse(sel.dataset.bucketFiles || "{}");
    const docs = sel.closest(".bucket-files-picker").querySelector(".bucket-files-docs");
    sel.onchange = () => {
      const files = map[sel.value] || [];
      docs.innerHTML = !sel.value ? ""
        : files.length ? files.map(f => `<li>📄 ${escapeHtml(f)}</li>`).join("")
        : `<li class="hint">empty</li>`;
    };
  });
}

// --- Collapsible chunk text (shared by every page that shows retrieved chunk bodies) ---

// Renders a chunk of retrieved text at full length inside a visually-clamped box. The full
// text is always in the DOM (never truncated), so nothing is lost; wireChunks() then reveals
// an Expand/Collapse toggle ONLY when the text actually overflows the clamp.
export function chunkTextHtml(text, extraClass = "") {
  return `<div class="chunk">
      <div class="chunk-text clamped${extraClass ? " " + extraClass : ""}">${escapeHtml(text ?? "")}</div>
      <button type="button" class="chunk-toggle" hidden>Expand</button>
    </div>`;
}

// Wires every .chunk inside `root`: measures overflow (needs the element to be laid out, i.e.
// not display:none) and, when clipped, shows a toggle that expands/collapses the clamp.
// Idempotent — safe to call again once a hidden container becomes visible.
export function wireChunks(root) {
  root.querySelectorAll(".chunk").forEach(chunk => {
    const body = chunk.querySelector(".chunk-text");
    const btn = chunk.querySelector(".chunk-toggle");
    if (!body || !btn) return;
    const overflowing = body.scrollHeight - body.clientHeight > 2;
    body.classList.toggle("overflowing", overflowing);
    btn.hidden = !overflowing;
    if (!overflowing) return;
    btn.onclick = () => {
      const clamped = body.classList.toggle("clamped");
      btn.textContent = clamped ? "Expand" : "Collapse";
    };
  });
}

export function getQaHistory() {
  return loadStore().qa || [];
}

export function addQaToHistory(entry) {
  const state = loadStore();
  state.qa = [entry, ...(state.qa || [])].slice(0, 20);
  saveStore(state);
}

// Wipes the local Q&A thread history (UI-only; does not touch indexed data).
export function clearQaHistory() {
  const state = loadStore();
  state.qa = [];
  saveStore(state);
}
