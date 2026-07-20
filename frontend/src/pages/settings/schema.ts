// Declarative field/section schema mirroring the .env layout (ui/pages/settings.js's SECTIONS,
// ported 1:1). Each field: key + input type; `pick` gives a dropdown's options; `secret` masks it;
// `req` marks a credential that must be non-empty while its block is active. Each gated section
// carries `gate(sel)` over the current provider selection; sections with no gate are always active.

export type FieldType = 'text' | 'password' | 'number' | 'bool' | 'select'

export interface GateSelection {
  llm_provider: string
  embedding_provider: string
  reranker_provider: string
  vector_store: string
}

export interface FieldDef {
  key: string
  type: FieldType
  pick?: string[]
  req?: boolean
  activeWhen?: (sel: GateSelection) => boolean
}

export interface SectionDef {
  id: string
  title: string
  control?: boolean
  note?: string
  gate?: (sel: GateSelection) => boolean
  fields: FieldDef[]
}

function T(key: string, type: FieldType = 'text', extra: Partial<FieldDef> = {}): FieldDef {
  return { key, type, ...extra }
}

export const GATE_KEYS: (keyof GateSelection)[] = ['llm_provider', 'embedding_provider', 'reranker_provider', 'vector_store']

// Per-field inline warnings for changes with data-migration consequences.
export const FIELD_WARN: Record<string, string> = {
  vector_store: 'Switching does not migrate data; each store holds its own indices. OpenSearch must be running (port 9201) when selected.',
  embedding_provider: 'Changing embeddings requires a NEW index — existing indices are locked to their build model & vector dim. Re-ingest on a fresh index after saving.',
  litellm_embedding_model: 'Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.',
  watsonx_embedding_model_id: 'Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.',
  fastembed_model_name: 'Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.',
  ollama_embedding_model: 'Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.',
  openai_embedding_model: 'Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.',
  openrouter_embedding_model: 'Changing the embedding model requires a NEW index (different vectors/dim). Re-ingest after saving.',
}

// Fields whose change invalidates existing indices -- a toast fires when the operator edits one.
export const REINDEX_KEYS = new Set(Object.keys(FIELD_WARN).filter((k) => k !== 'vector_store'))

export const SECTIONS: SectionDef[] = [
  {
    id: 'control',
    title: '§1 · Control panel',
    control: true,
    note: 'Provider selection — drives which blocks below are active.',
    fields: [
      T('llm_provider', 'select', { pick: ['watsonx', 'litellm', 'litellm-watsonx', 'litellm-ollama', 'litellm-openai', 'litellm-openrouter'] }),
      T('embedding_provider', 'select', {
        pick: ['watsonx', 'fastembed', 'litellm', 'litellm-watsonx', 'litellm-ollama', 'litellm-openai', 'litellm-openrouter'],
      }),
      T('reranker_provider', 'select', { pick: ['sentence-transformers', 'litellm', 'litellm-watsonx'] }),
      T('vector_store', 'select', { pick: ['opensearch', 'chroma'] }),
    ],
  },
  {
    id: 'watsonx',
    title: '§2 · watsonx backend',
    gate: (s) => [s.llm_provider, s.embedding_provider].some((p) => p === 'watsonx' || p === 'litellm-watsonx') || s.reranker_provider === 'litellm-watsonx',
    fields: [
      T('watsonx_url'),
      T('watsonx_apikey', 'password', { req: true }),
      T('watsonx_project_id', 'password', { req: true }),
      T('watsonx_llm_model_id'),
      T('watsonx_embedding_model_id'),
      T('watsonx_reranker_model_id'),
      T('watsonx_max_new_tokens', 'number'),
      T('watsonx_temperature', 'number'),
      T('watsonx_repetition_penalty', 'number'),
    ],
  },
  {
    id: 'litellm',
    title: '§3 · LiteLLM backend',
    gate: (s) => [s.llm_provider, s.embedding_provider, s.reranker_provider].includes('litellm'),
    note: 'Generic provider. litellm-watsonx/-ollama reuse §2/§4 instead. Per-modality fields fall back to the shared MODEL/API_BASE/API_KEY when empty.',
    fields: [
      T('litellm_model', 'text', { req: true }),
      T('litellm_api_base'),
      T('litellm_api_key', 'password'),
      T('litellm_max_new_tokens', 'number'),
      T('litellm_temperature', 'number'),
      T('litellm_embedding_model'),
      T('litellm_embedding_api_base'),
      T('litellm_embedding_api_key', 'password'),
      T('litellm_rerank_model'),
      T('litellm_rerank_api_base'),
      T('litellm_rerank_api_key', 'password'),
    ],
  },
  {
    id: 'ollama',
    title: '§4 · Ollama backend',
    gate: (s) => [s.llm_provider, s.embedding_provider].includes('litellm-ollama'),
    fields: [T('ollama_api_base'), T('ollama_model'), T('ollama_embedding_model')],
  },
  {
    id: 'openai',
    title: '§5 · OpenAI backend',
    gate: (s) => [s.llm_provider, s.embedding_provider].includes('litellm-openai'),
    fields: [
      T('openai_model', 'text', { req: true, activeWhen: (s) => s.llm_provider === 'litellm-openai' }),
      T('openai_api_key', 'password', { req: true, activeWhen: (s) => [s.llm_provider, s.embedding_provider].includes('litellm-openai') }),
      T('openai_api_base'),
      T('openai_max_new_tokens', 'number'),
      T('openai_temperature', 'number'),
      T('openai_embedding_model', 'text', { req: true, activeWhen: (s) => s.embedding_provider === 'litellm-openai' }),
    ],
  },
  {
    id: 'openrouter',
    title: '§6 · OpenRouter backend',
    gate: (s) => [s.llm_provider, s.embedding_provider].includes('litellm-openrouter'),
    fields: [
      T('openrouter_model', 'text', { req: true, activeWhen: (s) => s.llm_provider === 'litellm-openrouter' }),
      T('openrouter_api_key', 'password', { req: true, activeWhen: (s) => [s.llm_provider, s.embedding_provider].includes('litellm-openrouter') }),
      T('openrouter_api_base'),
      T('openrouter_max_new_tokens', 'number'),
      T('openrouter_temperature', 'number'),
      T('openrouter_embedding_model', 'text', { req: true, activeWhen: (s) => s.embedding_provider === 'litellm-openrouter' }),
    ],
  },
  {
    id: 'fastembed',
    title: '§7 · fastembed backend',
    gate: (s) => s.embedding_provider === 'fastembed',
    fields: [T('fastembed_model_name'), T('fastembed_cache_dir')],
  },
  {
    id: 'vs_chroma',
    title: '§8 · Vector store — Chroma',
    gate: (s) => s.vector_store === 'chroma',
    fields: [T('chroma_path')],
  },
  {
    id: 'vs_os',
    title: '§8 · Vector store — OpenSearch',
    gate: (s) => s.vector_store === 'opensearch',
    note: 'OpenSearch must be running (port 9201) when selected.',
    fields: [
      T('opensearch_url'),
      T('opensearch_user'),
      T('opensearch_password', 'password'),
      T('opensearch_verify_certs', 'bool'),
      T('opensearch_hybrid_pipeline'),
      T('hybrid_sparse_weight', 'number'),
      T('hybrid_dense_weight', 'number'),
    ],
  },
  {
    id: 'chunking',
    title: '§9 · Chunking',
    fields: [
      T('default_chunking_strategy', 'select', { pick: ['sentence', 'token', 'semantic', 'window'] }),
      T('chunk_size', 'number'),
      T('chunk_overlap', 'number'),
      T('semantic_breakpoint_percentile', 'number'),
      T('sentence_window_size', 'number'),
      T('llm_chunk_max_size', 'number'),
      T('llm_chunk_window_size', 'number'),
    ],
  },
  {
    id: 'retrieval',
    title: '§10 · Retrieval & rerank',
    fields: [
      T('retrieval_top_k', 'number'),
      T('rerank_top_n', 'number'),
      T('similarity_threshold', 'number'),
      T('reranker_model', 'text', { activeWhen: (s) => s.reranker_provider === 'sentence-transformers' }),
    ],
  },
  {
    id: 'generation',
    title: '§11 · Generation',
    fields: [T('citation_chunk_size', 'number'), T('prompts_dir'), T('default_prompt_name')],
  },
  { id: 'ingestion', title: '§12 · Ingestion', fields: [T('enrich_keywords', 'bool')] },
  { id: 'eval', title: '§13 · Eval harness', fields: [T('eval_dir')] },
  {
    id: 'logging',
    title: '§14 · Logging',
    fields: [T('log_level', 'select', { pick: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] })],
  },
]

export const KNOWN_KEYS = new Set(SECTIONS.flatMap((s) => s.fields.map((f) => f.key)))

export function selectionOf(cur: Record<string, unknown>): GateSelection {
  return {
    llm_provider: String(cur.llm_provider ?? ''),
    embedding_provider: String(cur.embedding_provider ?? ''),
    reranker_provider: String(cur.reranker_provider ?? ''),
    vector_store: String(cur.vector_store ?? ''),
  }
}
