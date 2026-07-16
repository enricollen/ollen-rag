// Shared helpers: same-origin fetch wrapper, toasts, localStorage-backed
// session state (job history, seen buckets), and small DOM/format utilities.

// Validated dark-mode categorical palette (dataviz skill's steps; this UI is dark-only). Fixed
// order, never cycled — shared by every chart so identity maps to the same hue across pages.
export const CHART_PALETTE = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];
export const CHART_GRID = "#2c2c2a"; // dataviz dark-mode gridline hairline

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

// Build the reranker <select> as one <optgroup> per provider. The provider rides a
// data-provider attribute rather than being packed into the option value: model ids contain "/",
// so any delimiter scheme would be ambiguous. Read it back with
// select.selectedOptions[0].dataset.provider.
//
// A provider with an empty model list is free-form (e.g. the generic "litellm" provider, which
// accepts any LiteLLM model string); it can only offer whatever model is configured for it, so it
// contributes an option only when reranker_default_models names one.
export function rerankerOptionsHtml(cfg) {
  const choices = cfg.reranker_model_choices || {};
  const defaults = cfg.reranker_default_models || {};
  return Object.entries(choices).map(([provider, models]) => {
    const options = (models && models.length ? models : [defaults[provider]].filter(Boolean));
    if (!options.length) return "";
    const items = options.map(model => {
      const selected = provider === cfg.reranker_provider && model === cfg.reranker_model;
      return `<option value="${escapeHtml(model)}" data-provider="${escapeHtml(provider)}"${selected ? " selected" : ""}>${escapeHtml(model)}</option>`;
    }).join("");
    return `<optgroup label="${escapeHtml(provider)}">${items}</optgroup>`;
  }).join("");
}

// Read both halves of a reranker <select> back out: the model is the option value, the provider
// is on the selected option's dataset.
export function rerankerSelection(selectId) {
  const select = document.getElementById(selectId);
  const option = select?.selectedOptions?.[0];
  return { reranker_provider: option?.dataset.provider || null, reranker_model: select?.value || null };
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
export function chunkingSummary(chunking) {
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

// --- Active-config banner (top of every page) -------------------------------------------------
// Compact read-out of the live wiring the whole console runs on, sourced from /api/v1/config's
// `active` block (same resolution as the startup log). Purely informational.
export function activeBannerHtml(active) {
  if (!active) return "";
  const item = (ico, label, val) =>
    `<span class="ab-item"><span class="ab-ico">${ico}</span><span class="ab-lbl">${escapeHtml(label)}</span><span class="ab-val">${escapeHtml(val)}</span></span>`;
  return `
    <div class="active-banner" title="Active configuration — change it in Settings">
      <div class="ab-title">⚙️ Active Configuration</div>
      <div class="ab-items">
        ${item("💬", "LLM", `${active.llm.provider} · ${active.llm.model}`)}
        ${item("🧮", "Embedding", `${active.embedding.provider} · ${active.embedding.model}`)}
        ${item("🎯", "Reranker", `${active.reranker.provider} · ${active.reranker.model}`)}
        ${item("🗄️", "Vector store", active.vector_store)}
        ${item("✂️", "Chunking", `${active.chunking.strategy} · ${active.chunking.chunk_size}/${active.chunking.chunk_overlap}`)}
        ${item("📊", "top_k / top_n", `${active.retrieval_top_k} / ${active.rerank_top_n}`)}
      </div>
    </div>`;
}

// --- Cross-store "existing indexes" panel (shared by Ingestion + Indices) ----------------------
// One group per vector store (from /api/v1/indices/overview), each a grid of index cards showing
// build config + bucket chips. `opts.deletable` adds a 🗑 button on active-store cards;
// `opts.selectable` marks active-store cards clickable (inactive-store indexes can't be browsed or
// deleted — those endpoints target the active backend only). Wire interactions with
// wireIndicesOverview(); bucket chips reveal their file list without extra wiring here.
export function indexOverviewCardHtml(store, ix, { deletable = false, selectable = false, active = true } = {}) {
  const emb = ix.embedding_provider ? `${ix.embedding_provider}/${ix.embedding_model}` : "unrecorded";
  const buckets = Object.keys(ix.bucket_files || {});
  const chips = buckets.length
    ? buckets.map(b => `<span class="pill kb-bucket" data-store="${escapeHtml(store)}" data-index="${escapeHtml(ix.index)}" data-bucket="${escapeHtml(b)}">📦 ${escapeHtml(b)} <span class="kb-count">${(ix.bucket_files[b] || []).length}</span></span>`).join("")
    : `<span class="hint">no buckets yet</span>`;
  const manageable = active;  // browse/delete only work against the active backend
  const cls = ["kb-index", selectable && manageable ? "kb-index--select" : "", !manageable ? "kb-index--locked" : ""].filter(Boolean).join(" ");
  const delBtn = deletable && manageable
    ? `<button type="button" class="kb-del" data-store="${escapeHtml(store)}" data-index="${escapeHtml(ix.index)}" title="Delete this index">🗑 Delete</button>`
    : "";
  return `
    <div class="${cls}" data-store="${escapeHtml(store)}" data-index="${escapeHtml(ix.index)}">
      <div class="kb-index-head">
        <code class="inline">${escapeHtml(ix.index)}</code>
        <span class="kb-head-right"><span class="kb-docs">${ix.docs_count} docs</span>${delBtn}</span>
      </div>
      <div class="kb-index-meta">
        <span class="pill pill-soft">🧮 ${escapeHtml(emb)}</span>
        <span class="pill pill-soft">✂️ ${escapeHtml(chunkingSummary(ix.chunking))}</span>
      </div>
      <div class="kb-buckets">${chips}</div>
      <div class="kb-bucket-files" data-files-host></div>
      ${!manageable ? `<div class="hint kb-locked-note">inactive store — switch VECTOR_STORE in Settings to browse or delete</div>` : ""}</div>`;
}

export function indexOverviewHtml(overview, opts = {}) {
  if (!overview || !Array.isArray(overview.stores)) return "";
  const stores = overview.stores.map(st => {
    const badge = st.active ? '<span class="pill pill-ok">active</span>' : '<span class="pill pill-soft">inactive</span>';
    let body;
    if (!st.available) body = `<div class="info-warn">⚠ unavailable — ${escapeHtml(st.error || "cannot reach this store")}</div>`;
    else if (!st.indices.length) body = `<div class="hint" style="padding:.4rem 0">No indexes in this store yet.</div>`;
    else body = `<div class="kb-index-grid">${st.indices.map(ix => indexOverviewCardHtml(st.vector_store, ix, { ...opts, active: st.active })).join("")}</div>`;
    return `
      <div class="kb-store">
        <div class="kb-store-head"><span class="kb-store-name">🗄️ ${escapeHtml(st.vector_store)}</span>${badge}</div>
        ${body}
      </div>`;
  }).join("");
  return stores;
}

// Wires bucket-chip toggles inside any container holding overview cards. `overview` is the source
// data (chip -> its files) so no refetch is needed. Optional `onSelect(store, index, cardEl)` fires
// when a selectable card body is clicked (Indices page uses it to drive the browse panels).
export function wireIndicesOverview(root, overview, { onSelect, onDelete } = {}) {
  root.querySelectorAll(".kb-bucket").forEach(chip => {
    chip.onclick = (e) => {
      e.stopPropagation();  // don't also trigger card-select
      const card = chip.closest(".kb-index");
      const host = card.querySelector("[data-files-host]");
      const reopen = !chip.classList.contains("selected");
      card.querySelectorAll(".kb-bucket").forEach(c => c.classList.remove("selected"));
      if (!reopen) { host.innerHTML = ""; return; }
      chip.classList.add("selected");
      const store = overview?.stores.find(s => s.vector_store === chip.dataset.store);
      const ix = store?.indices.find(i => i.index === chip.dataset.index);
      const files = ix?.bucket_files?.[chip.dataset.bucket];
      host.innerHTML = !files || !files.length
        ? `<div class="hint" style="padding:.2rem 0">empty bucket — no documents</div>`
        : `<ul class="kb-file-list">${files.map(f => `<li>📄 ${escapeHtml(f)}</li>`).join("")}</ul>`;
    };
  });
  if (onSelect) root.querySelectorAll(".kb-index--select").forEach(card => {
    card.onclick = () => onSelect(card.dataset.store, card.dataset.index, card);
  });
  if (onDelete) root.querySelectorAll(".kb-del").forEach(btn => {
    btn.onclick = (e) => { e.stopPropagation(); onDelete(btn.dataset.store, btn.dataset.index); };
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
