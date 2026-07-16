// Settings page: an editable mirror of .env, laid out in the same numbered modules so the operator
// immediately sees which providers are active, which blocks are inert, and which credentials still
// need a value. Sourced from GET /api/v1/settings; Save writes .env via POST /api/v1/settings and
// restarts the dev service, then the page polls /health until the worker is back and reloads.
import { api, errorMessage, escapeHtml, toast } from "../lib.js";

// Per-field inline warnings for changes with data-migration consequences.
const FIELD_WARN = {
  vector_store: "Switching does not migrate data; each store holds its own indices. OpenSearch must be running (port 9201) when selected.",
  embedding_provider: "Changing embeddings requires a NEW index — existing indices are locked to their build model & vector dim. Re-ingest on a fresh index after saving.",
  litellm_embedding_model: "Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.",
  watsonx_embedding_model_id: "Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.",
  fastembed_model_name: "Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.",
  ollama_embedding_model: "Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.",
  openai_embedding_model: "Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.",
  openrouter_embedding_model: "Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.",
};
// Fields whose change invalidates existing indices — a toast fires when the operator edits one.
const REINDEX_KEYS = new Set(["embedding_provider", "litellm_embedding_model", "watsonx_embedding_model_id", "fastembed_model_name", "ollama_embedding_model", "openai_embedding_model", "openrouter_embedding_model"]);

// The four provider switches. Changing any of them re-evaluates every block's ACTIVE gate live.
const GATE_KEYS = ["llm_provider", "embedding_provider", "reranker_provider", "vector_store"];

// Declarative field/section schema mirroring the .env layout. Each field: key + input type; `pick`
// gives a dropdown's options; `secret` masks it; `req` marks a credential that must be non-empty
// while its block is active (renders a "needs value" flag). Each gated section carries `gate(sel)`
// over the current {llm_provider, embedding_provider, reranker_provider, vector_store} selection;
// sections with no gate are always active.
const T = (key, type = "text", extra = {}) => ({ key, type, ...extra });

const SECTIONS = [
  {
    id: "control", title: "§1 · Control panel", control: true,
    note: "Provider selection — drives which blocks below are active.",
    fields: [
      T("llm_provider", "select", { pick: ["watsonx", "litellm", "litellm-watsonx", "litellm-ollama", "litellm-openai", "litellm-openrouter"] }),
      T("embedding_provider", "select", { pick: ["watsonx", "fastembed", "litellm", "litellm-watsonx", "litellm-ollama", "litellm-openai", "litellm-openrouter"] }),
      T("reranker_provider", "select", { pick: ["sentence-transformers", "litellm", "litellm-watsonx"] }),
      T("vector_store", "select", { pick: ["opensearch", "chroma"] }),
    ],
  },
  {
    id: "watsonx", title: "§2 · watsonx backend",
    gate: s => [s.llm_provider, s.embedding_provider].some(p => p === "watsonx" || p === "litellm-watsonx") || s.reranker_provider === "litellm-watsonx",
    fields: [
      T("watsonx_url"),
      T("watsonx_apikey", "password", { req: true }),
      T("watsonx_project_id", "password", { req: true }),
      T("watsonx_llm_model_id"), T("watsonx_embedding_model_id"), T("watsonx_reranker_model_id"),
      T("watsonx_max_new_tokens", "number"), T("watsonx_temperature", "number"), T("watsonx_repetition_penalty", "number"),
    ],
  },
  {
    id: "litellm", title: "§3 · LiteLLM backend",
    gate: s => [s.llm_provider, s.embedding_provider, s.reranker_provider].includes("litellm"),
    note: "Generic provider. litellm-watsonx/-ollama reuse §2/§4 instead. Per-modality fields fall back to the shared MODEL/API_BASE/API_KEY when empty.",
    fields: [
      T("litellm_model", "text", { req: true }), T("litellm_api_base"), T("litellm_api_key", "password"),
      T("litellm_max_new_tokens", "number"), T("litellm_temperature", "number"),
      T("litellm_embedding_model"), T("litellm_embedding_api_base"), T("litellm_embedding_api_key", "password"),
      T("litellm_rerank_model"), T("litellm_rerank_api_base"), T("litellm_rerank_api_key", "password"),
    ],
  },
  {
    id: "ollama", title: "§4 · Ollama backend",
    gate: s => [s.llm_provider, s.embedding_provider].includes("litellm-ollama"),
    fields: [T("ollama_api_base"), T("ollama_model"), T("ollama_embedding_model")],
  },
  {
    id: "openai", title: "§5 · OpenAI backend",
    gate: s => [s.llm_provider, s.embedding_provider].includes("litellm-openai"),
    fields: [
      T("openai_model"), T("openai_api_key", "password"), T("openai_api_base"),
      T("openai_max_new_tokens", "number"), T("openai_temperature", "number"),
      T("openai_embedding_model"),
    ],
  },
  {
    id: "openrouter", title: "§6 · OpenRouter backend",
    gate: s => [s.llm_provider, s.embedding_provider].includes("litellm-openrouter"),
    fields: [
      T("openrouter_model"), T("openrouter_api_key", "password"), T("openrouter_api_base"),
      T("openrouter_max_new_tokens", "number"), T("openrouter_temperature", "number"),
      T("openrouter_embedding_model"),
    ],
  },
  {
    id: "fastembed", title: "§7 · fastembed backend",
    gate: s => s.embedding_provider === "fastembed",
    fields: [T("fastembed_model_name"), T("fastembed_cache_dir")],
  },
  {
    id: "vs_chroma", title: "§8 · Vector store — Chroma",
    gate: s => s.vector_store === "chroma",
    fields: [T("chroma_path")],
  },
  {
    id: "vs_os", title: "§8 · Vector store — OpenSearch",
    gate: s => s.vector_store === "opensearch",
    note: "OpenSearch must be running (port 9201) when selected.",
    fields: [
      T("opensearch_url"), T("opensearch_user"), T("opensearch_password", "password"),
      T("opensearch_verify_certs", "bool"), T("opensearch_hybrid_pipeline"),
      T("hybrid_sparse_weight", "number"), T("hybrid_dense_weight", "number"),
    ],
  },
  {
    id: "chunking", title: "§9 · Chunking",
    fields: [
      T("default_chunking_strategy", "select", { pick: ["sentence", "token", "semantic", "window"] }),
      T("chunk_size", "number"), T("chunk_overlap", "number"), T("semantic_breakpoint_percentile", "number"),
      T("sentence_window_size", "number"), T("llm_chunk_max_size", "number"), T("llm_chunk_window_size", "number"),
    ],
  },
  {
    id: "retrieval", title: "§10 · Retrieval & rerank",
    fields: [
      T("retrieval_top_k", "number"), T("rerank_top_n", "number"), T("similarity_threshold", "number"),
      // Only used when reranker_provider = sentence-transformers; flagged inline rather than hidden.
      T("reranker_model", "text", { activeWhen: s => s.reranker_provider === "sentence-transformers" }),
    ],
  },
  {
    id: "generation", title: "§11 · Generation",
    fields: [T("citation_chunk_size", "number"), T("prompts_dir"), T("default_prompt_name")],
  },
  { id: "ingestion", title: "§12 · Ingestion", fields: [T("enrich_keywords", "bool")] },
  { id: "eval", title: "§13 · Eval harness", fields: [T("eval_dir")] },
  {
    id: "logging", title: "§14 · Logging",
    fields: [T("log_level", "select", { pick: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] })],
  },
];

// Every field the schema knows about, for detecting server keys we haven't mapped (future drift).
const KNOWN_KEYS = new Set(SECTIONS.flatMap(s => s.fields.map(f => f.key)));

// Current selection object the gates read.
const selectionOf = cur => Object.fromEntries(GATE_KEYS.map(k => [k, cur[k]]));

// Render one field's input, coercing the current value to a string for the DOM.
function inputHtml(field, val) {
  const id = `f-${field.key}`;
  const v = val ?? "";
  if (field.type === "select") {
    return `<select id="${id}" data-key="${field.key}">${field.pick.map(o => `<option${o === val ? " selected" : ""}>${escapeHtml(o)}</option>`).join("")}</select>`;
  }
  if (field.type === "bool") {
    return `<select id="${id}" data-key="${field.key}"><option value="true"${val ? " selected" : ""}>true</option><option value="false"${!val ? " selected" : ""}>false</option></select>`;
  }
  const type = field.type === "password" ? "password" : field.type === "number" ? "number" : "text";
  const step = field.type === "number" ? ' step="any"' : "";
  return `<input id="${id}" data-key="${field.key}" type="${type}"${step} value="${escapeHtml(v)}" />`;
}

// Render one field row: key label (+ flags) on the left, input on the right.
function fieldRowHtml(field, cur, sectionActive) {
  const val = cur[field.key];
  const isEmpty = val === "" || val == null;
  // "needs value": a required credential, its block active, still empty.
  const needsValue = field.req && sectionActive && isEmpty;
  // A field disabled by an intra-section gate (e.g. reranker_model when not sentence-transformers).
  const fieldInert = field.activeWhen && !field.activeWhen(selectionOf(cur));
  const flags = [
    needsValue ? `<span class="pill pill-warn">needs value</span>` : "",
    fieldInert ? `<span class="pill">inactive</span>` : "",
  ].join("");
  const warn = FIELD_WARN[field.key]
    ? `<div class="hint" style="color:var(--warn)">⚠ ${escapeHtml(FIELD_WARN[field.key])}</div>` : "";
  return `<div class="kv-row"${fieldInert ? ' style="opacity:.5"' : ""}>
      <span class="k">${escapeHtml(field.key)} ${flags}</span>
      <span class="v" style="flex:1;max-width:60%">${inputHtml(field, val)}${warn}</span>
    </div>`;
}

// Render one section card with an active/inactive badge and its fields.
function sectionHtml(section, cur) {
  const active = !section.gate || section.gate(selectionOf(cur));
  const badge = section.control
    ? ""
    : active ? `<span class="pill pill-ok">active</span>` : `<span class="pill">inactive</span>`;
  const rows = section.fields.map(f => fieldRowHtml(f, cur, active)).join("");
  const note = section.note ? `<div class="hint" style="margin:-.2rem 0 .7rem">${escapeHtml(section.note)}</div>` : "";
  const vsWarn = section.id === "vs_chroma" || section.id === "vs_os"
    ? `<div class="hint">Switching vector store does not migrate data — each store holds its own indices.</div>` : "";
  return `<div class="card${active ? "" : " settings-inactive"}"${section.control ? ' style="border-color:var(--accent)"' : ""}>
      <h2>${escapeHtml(section.title)} ${badge}</h2>
      ${note}<div class="kv-list">${rows}</div>${vsWarn}
    </div>`;
}

// The "current wiring" banner: plain-language summary of what the switches resolve to.
function wiringHtml(cur) {
  const reuse = { "litellm-watsonx": " (reuses watsonx §2)", "litellm-ollama": " (reuses ollama §4)" };
  const line = (label, val, extra = "") => `<span class="pill pill-ok">${label}: ${escapeHtml(val)}${extra}</span>`;
  return `<div class="chip-row" style="margin-bottom:1.1rem">
      ${line("LLM", cur.llm_provider, reuse[cur.llm_provider] || "")}
      ${line("Embeddings", cur.embedding_provider, reuse[cur.embedding_provider] || "")}
      ${line("Reranker", cur.reranker_provider)}
      ${line("Vectors", cur.vector_store)}
    </div>`;
}

export async function render(view) {
  view.innerHTML = `
    <h1 class="page-title">Settings</h1>
    <p class="page-sub">Editable mirror of <code class="inline">.env</code>, grouped by module. Inactive blocks are dimmed; <span class="pill pill-warn">needs value</span> marks required credentials still empty. Save writes <code class="inline">.env</code> and restarts the service.</p>
    <div id="settings-body"><div class="empty-state"><span class="spinner"></span></div></div>
  `;
  const body = document.getElementById("settings-body");
  let initial;
  try {
    initial = await api("/api/v1/settings");
  } catch (e) {
    body.innerHTML = `<div class="card">Could not load settings: ${errorMessage(e)}</div>`;
    return;
  }
  const current = { ...initial };

  // Any server field not covered by SECTIONS gets a catch-all block so nothing is hidden/unsavable.
  const unmapped = Object.keys(initial).filter(k => !KNOWN_KEYS.has(k)).sort();
  const sections = unmapped.length
    ? [...SECTIONS, { id: "other", title: "Other (unmapped)", fields: unmapped.map(k => T(k, typeof initial[k] === "number" ? "number" : typeof initial[k] === "boolean" ? "bool" : "text")) }]
    : SECTIONS;

  // Re-read a DOM input into a typed value matching the initial value's type.
  const readInput = (el, key) => {
    let v = el.value;
    if (typeof initial[key] === "number") v = Number(v);
    else if (typeof initial[key] === "boolean") v = v === "true";
    return v;
  };

  // Paint the whole form from `current`, then wire listeners. Re-called on any gate/required change
  // so badges, dimming and the wiring banner stay live without a save.
  function paint() {
    body.innerHTML = `
      ${wiringHtml(current)}
      <div class="grid-2">${sections.map(s => sectionHtml(s, current)).join("")}</div>
      <div style="margin-top:1.2rem;display:flex;gap:1rem;align-items:center">
        <button id="save-settings" class="primary">Save &amp; restart</button>
        <span id="save-status" class="hint"></span>
      </div>
    `;
    body.querySelectorAll("[data-key]").forEach(el => {
      const key = el.dataset.key;
      el.addEventListener("change", () => {
        current[key] = readInput(el, key);
        // Editing an embedding-affecting field invalidates existing indices — warn on the spot.
        if (REINDEX_KEYS.has(key) && current[key] !== initial[key]) {
          toast("Embedding changed — create a NEW index and re-ingest; existing indices are locked to their build model.", "error");
        }
        // A switch or required field changed — repaint so gates/flags refresh (change fires on blur,
        // so no focus loss mid-typing).
        if (GATE_KEYS.includes(key) || sections.some(s => s.fields.some(f => f.key === key && f.req))) paint();
      });
      // Keep `current` live on every keystroke too, so Save captures in-progress edits.
      el.addEventListener("input", () => { current[key] = readInput(el, key); });
    });
    document.getElementById("save-settings").addEventListener("click", onSave);
  }

  async function onSave() {
    const status = document.getElementById("save-status");
    const changes = {};
    for (const key of Object.keys(current)) if (current[key] !== initial[key]) changes[key] = current[key];
    if (Object.keys(changes).length === 0) { status.textContent = "No changes."; return; }
    document.getElementById("save-settings").disabled = true;
    status.textContent = "Saving…";
    let res;
    try {
      res = await api("/api/v1/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
    } catch (e) {
      status.textContent = `Save failed: ${errorMessage(e)}`;
      document.getElementById("save-settings").disabled = false;
      return;
    }
    if (!res.restarting) {
      // "manual" mode: .env is written, but nothing supervises this process to reload it
      // (no --reload, no container restart policy) -- apply_restart() is a deliberate no-op
      // rather than kill a process nothing will bring back. Reloading now would just re-show
      // the same stale in-memory Settings, so don't pretend a restart happened.
      status.textContent = "Saved to .env — restart the service manually to apply (restart_mode=manual, no supervisor detected).";
      document.getElementById("save-settings").disabled = false;
      return;
    }
    status.textContent = "Restarting service…";
    waitForRestart(status);
  }

  paint();
}

// Poll /health until it returns ok or the deadline passes, then reload the page.
async function waitForRestart(statusEl) {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 1000));
    try {
      const h = await api("/health");
      if (h.status === "ok") { location.reload(); return; }
    } catch (_) { /* worker still down, keep polling */ }
  }
  statusEl.textContent = "Service did not come back within 30s — check the server, then reload.";
}