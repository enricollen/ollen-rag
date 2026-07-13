// First-run setup wizard: pick an LLM provider, (optionally) enter credentials, test them live,
// choose local embeddings/rerank + vector store, then write .env and restart. Reuses the same
// /api/v1/settings write path as the Settings page — this is guided config, not a separate system.
import { api, toast, errorMessage } from "../lib.js";

// The provider choices offered in step 1, with the settings each one needs.
const LLM_CHOICES = [
  { id: "litellm-ollama", label: "Local (Ollama) — no key, runs here", fields: [] },
  { id: "watsonx", label: "watsonx.ai", fields: ["watsonx_url", "watsonx_apikey", "watsonx_project_id"] },
  { id: "litellm", label: "OpenAI / other (LiteLLM)", fields: ["litellm_model", "litellm_api_key", "litellm_api_base"] },
];

// Renders the wizard into `view`. Same signature as every other page module.
export async function render(view) {
  // Local wizard state; nothing is persisted until Finish.
  const state = { llm: "litellm-ollama", creds: {}, store: "chroma" };

  // Read-only compute indicator: cpu/gpu is baked at image build (TORCH_FLAVOR), not switchable
  // here, so we only report what was built. Failure to read it just hides the line.
  let compute = "";
  try { compute = (await api("/api/v1/onboarding/status")).compute || ""; } catch { /* hide */ }
  const computeNote = compute
    ? `<p class="compute-note">Compute: <strong>${compute.toUpperCase()}</strong>${compute === "cpu" ? " — rebuild with <code>TORCH_FLAVOR=gpu</code> for GPU" : ""}</p>`
    : "";

  // Step 1: choose how answers get generated.
  function stepLLM() {
    view.innerHTML = `
      <div class="card">
        <h2>Welcome — set up ollen-rag</h2>
        <p>Choose how answers get generated. Local needs no account.</p>
        <div id="llm-opts"></div>
        <button id="next" class="btn primary">Next</button>
        ${computeNote}
      </div>`;
    const opts = view.querySelector("#llm-opts");
    opts.innerHTML = LLM_CHOICES.map(c =>
      `<label class="radio"><input type="radio" name="llm" value="${c.id}" ${c.id === state.llm ? "checked" : ""}> ${c.label}</label>`
    ).join("");
    view.querySelector("#next").onclick = () => {
      state.llm = view.querySelector('input[name="llm"]:checked').value;
      const choice = LLM_CHOICES.find(c => c.id === state.llm);
      choice.fields.length ? stepCreds(choice) : stepStore();  // Ollama skips credentials
    };
  }

  // Step 2 (cloud providers only): enter + live-test credentials before continuing.
  function stepCreds(choice) {
    view.innerHTML = `
      <div class="card">
        <h2>${choice.label} credentials</h2>
        <div id="fields">${choice.fields.map(f =>
          `<label>${f}<input data-field="${f}" type="${f.includes("key") ? "password" : "text"}" value="${state.creds[f] || ""}"></label>`
        ).join("")}</div>
        <div id="test-result"></div>
        <button id="test" class="btn">Test connection</button>
        <button id="next" class="btn primary" disabled>Next</button>
        <button id="back" class="btn">Back</button>
      </div>`;
    const readFields = () => choice.fields.forEach(f => { state.creds[f] = view.querySelector(`[data-field="${f}"]`).value; });
    view.querySelector("#back").onclick = stepLLM;
    view.querySelector("#test").onclick = async () => {
      readFields();
      const changes = { llm_provider: state.llm, ...state.creds };
      try {
        const r = await api("/api/v1/onboarding/test", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: "llm", changes }) });
        const box = view.querySelector("#test-result");
        box.textContent = r.ok ? "✅ Connected" : `❌ ${r.detail}`;
        view.querySelector("#next").disabled = !r.ok;
      } catch (e) { toast(errorMessage(e), "error"); }
    };
    view.querySelector("#next").onclick = () => { readFields(); stepStore(); };
  }

  // Step 3: pick the vector store (Chroma needs nothing else running).
  function stepStore() {
    view.innerHTML = `
      <div class="card">
        <h2>Vector store</h2>
        <label class="radio"><input type="radio" name="store" value="chroma" ${state.store === "chroma" ? "checked" : ""}> Chroma — on-disk, nothing else to run (recommended)</label>
        <label class="radio"><input type="radio" name="store" value="opensearch"> OpenSearch — needs the opensearch compose profile running</label>
        <button id="finish" class="btn primary">Finish</button>
        <button id="back" class="btn">Back</button>
      </div>`;
    view.querySelector("#back").onclick = stepLLM;
    view.querySelector("#finish").onclick = () => { state.store = view.querySelector('input[name="store"]:checked').value; finish(); };
  }

  // Step 4: persist everything through the same /api/v1/settings write path, then wait for restart.
  async function finish() {
    // Embeddings + reranker default to keyless local; store + LLM come from wizard state.
    const changes = {
      llm_provider: state.llm, embedding_provider: "fastembed",
      reranker_provider: "sentence-transformers", vector_store: state.store, ...state.creds,
    };
    view.innerHTML = `<div class="card"><h2>Saving…</h2><span class="spinner"></span></div>`;
    try {
      const res = await api("/api/v1/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes) });
      if (res.restart_mode === "manual") {
        view.innerHTML = `<div class="card"><h2>Saved ✅</h2><p>Config written. Restart the service to apply, then reload this page.</p></div>`;
        return;
      }
      await waitForHealth();
      location.hash = "#/query";
    } catch (e) { view.innerHTML = `<div class="card"><h2>Save failed</h2><p>${errorMessage(e)}</p></div>`; }
  }

  // Poll /health until the restarted worker answers, then let the console load.
  async function waitForHealth() {
    for (let i = 0; i < 60; i++) {
      try { await api("/health"); if (i > 0) return; } catch { /* still restarting */ }
      await new Promise(r => setTimeout(r, 1000));
    }
  }

  stepLLM();
}
