// Settings page: read-only view of the service's effective (non-secret)
// configuration, sourced from GET /api/v1/config. Editing requires changing
// .env and restarting the service — there is no write path by design.
import { api, errorMessage } from "../lib.js";

function kv(k, v, mono = false) {
  return `<div class="kv-row"><span class="k">${k}</span><span class="v${mono ? " mono" : ""}">${v}</span></div>`;
}

export async function render(view) {
  view.innerHTML = `
    <h1 class="page-title">Settings</h1>
    <p class="page-sub">Effective configuration read from the server's environment (OLLEN_RAG_*). Read-only here — edit <code class="inline">.env</code> and restart the service to change it.</p>
    <div id="settings-body"><div class="empty-state"><span class="spinner"></span></div></div>
  `;
  const body = document.getElementById("settings-body");
  let cfg;
  try {
    cfg = await api("/api/v1/config");
  } catch (e) {
    body.innerHTML = `<div class="card">Could not load config: ${errorMessage(e)}</div>`;
    return;
  }

  const sparsePct = Math.round(cfg.hybrid_sparse_weight * 100);
  const densePct = 100 - sparsePct;

  body.innerHTML = `
    <div class="grid-2">
      <div class="card">
        <h2>Model providers</h2>
        <div class="kv-list">
          ${kv("Embedding provider", `<span class="pill">${cfg.embedding_provider}</span>`)}
          ${cfg.embedding_provider === "watsonx"
            ? kv("Watsonx embedding model", cfg.watsonx_embedding_model_id, true)
            : kv("Fastembed model", cfg.fastembed_model_name, true)}
          ${kv("LLM provider", `<span class="pill">${cfg.llm_provider}</span>`)}
          ${kv("Watsonx LLM model", cfg.watsonx_llm_model_id, true)}
        </div>
      </div>

      <div class="card">
        <h2>OpenSearch</h2>
        <div class="kv-list">
          ${kv("URL", cfg.opensearch_url, true)}
          ${kv("Index prefix", cfg.opensearch_index_prefix, true)}
          ${kv("Hybrid pipeline", cfg.opensearch_hybrid_pipeline, true)}
        </div>
        <div class="hint" style="margin-top:.75rem">Hybrid score weights (sparse BM25 vs. dense kNN)</div>
        <div class="weight-bar"><div class="sparse" style="width:${sparsePct}%"></div><div class="dense" style="width:${densePct}%"></div></div>
        <div class="weight-legend"><span>sparse ${cfg.hybrid_sparse_weight}</span><span>dense ${cfg.hybrid_dense_weight}</span></div>
      </div>

      <div class="card">
        <h2>Chunking defaults</h2>
        <div class="kv-list">
          ${kv("Default strategy", `<span class="pill">${cfg.default_chunking_strategy}</span>`)}
          ${kv("Chunk size", cfg.chunk_size)}
          ${kv("Chunk overlap", cfg.chunk_overlap)}
          ${kv("Semantic breakpoint percentile", cfg.semantic_breakpoint_percentile)}
          ${kv("Sentence window size", cfg.sentence_window_size)}
          ${kv("LLM chunk max size", cfg.llm_chunk_max_size)}
          ${kv("LLM chunk window size", cfg.llm_chunk_window_size)}
        </div>
      </div>

      <div class="card">
        <h2>Retrieval &amp; generation</h2>
        <div class="kv-list">
          ${kv("Retrieval top_k", cfg.retrieval_top_k)}
          ${kv("Rerank top_n", cfg.rerank_top_n)}
          ${kv("Similarity threshold", cfg.similarity_threshold != null ? cfg.similarity_threshold : '<span class="hint">not set (disabled)</span>')}
          ${kv("Reranker model", cfg.reranker_model, true)}
          ${kv("Citation chunk size", cfg.citation_chunk_size)}
          ${kv("Default prompt", cfg.default_prompt_name, true)}
        </div>
      </div>
    </div>
  `;
}
