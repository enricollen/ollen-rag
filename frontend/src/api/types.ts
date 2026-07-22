// Shapes mirror src/api/routes.py and src/api/onboarding.py response/request bodies exactly.

export interface FilterSpec {
  key: string
  value: unknown
  operator: string
}

export interface RetrieveParams {
  query: string
  strategy?: string | null
  index_name?: string | null
  top_k?: number | null
  rerank_top_n?: number | null
  similarity_threshold?: number | null
  filters?: FilterSpec[] | null
  filter_condition?: 'and' | 'or'
  reranker_provider?: string | null
  reranker_model?: string | null
}

export interface QueryParams extends RetrieveParams {
  prompt_name?: string | null
}

export interface RetrievedNode {
  text: string
  score: number | null
  metadata: Record<string, unknown>
  retrieval_score?: number | null
}

export interface RetrieveResponse {
  nodes: RetrievedNode[]
  hybrid_nodes: RetrievedNode[]
  bm25_nodes: RetrievedNode[]
  dense_nodes: RetrievedNode[]
}

export interface QuerySource extends RetrievedNode {
  id: number | string
}

export interface QueryResponse {
  answer: string
  sources: QuerySource[]
}

export interface IngestJobStatus {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  detail?: string | null
  result?: {
    num_documents: number
    num_nodes: number
    index: string
    skipped_duplicate?: boolean
    duplicate_of?: string
  } | null
  progress?: number
  stage?: string
}

export interface ChunkingConfig {
  strategy: string
  [knob: string]: string | number
}

export interface ActiveComponentSummary {
  llm: { provider: string; model: string }
  embedding: { provider: string; model: string }
  reranker: { provider: string; model: string }
  vector_store: string
  chunking: { strategy: string; chunk_size: number; chunk_overlap: number }
  retrieval_top_k: number
  rerank_top_n: number
}

export interface ConfigResponse {
  active: ActiveComponentSummary
  vector_store: string
  embedding_provider: string
  llm_provider: string
  watsonx_llm_model_id: string
  watsonx_repetition_penalty: number
  watsonx_embedding_model_id: string
  fastembed_model_name: string
  embedding_model_choices: Record<string, string[]>
  embedding_default_models: Record<string, string>
  opensearch_url: string
  opensearch_hybrid_pipeline: string
  hybrid_sparse_weight: number
  hybrid_dense_weight: number
  default_chunking_strategy: string
  enrich_keywords: boolean
  log_level: string
  chunk_size: number
  chunk_overlap: number
  semantic_breakpoint_percentile: number
  sentence_window_size: number
  llm_chunk_max_size: number
  llm_chunk_window_size: number
  retrieval_top_k: number
  similarity_threshold: number
  rerank_top_n: number
  reranker_provider: string
  reranker_model: string
  reranker_model_choices: Record<string, string[]>
  reranker_default_models: Record<string, string>
  citation_chunk_size: number
  default_prompt_name: string
}

// GET/POST /api/v1/settings -- full Settings.model_dump(), secrets included. Loosely typed since
// the form is schema-driven (see pages/settings) rather than exhaustively modeled here.
export type SettingsDump = Record<string, string | number | boolean | null>

export interface IndexListEntry {
  index: string
  'docs.count': number | string
  [key: string]: unknown
}

export interface IndexOverviewEntry {
  index: string
  docs_count: number
  embedding_provider: string | null
  embedding_model: string | null
  chunking: ChunkingConfig | null
  bucket_files: Record<string, string[]>
}

export interface IndexOverviewStore {
  vector_store: string
  active: boolean
  available: boolean
  indices: IndexOverviewEntry[]
  error?: string
}

export interface IndicesOverviewResponse {
  active_vector_store: string
  stores: IndexOverviewStore[]
}

export interface IndexInfo {
  index: string
  embedding_provider: string | null
  embedding_model: string | null
  chunking: ChunkingConfig | null
  dim: number | null
  docs_count: number | null
  buckets: string[]
  bucket_files: Record<string, string[]>
  unbucketed_files: string[]
}

export interface IndexDocument {
  id: string
  content: string
  metadata: Record<string, unknown>
}

export interface IndexDocumentsResponse {
  documents: IndexDocument[]
  total: number
}

export interface IndexVectorPoint {
  id: string
  x: number
  y: number
  bucket: string | null
  file_name: string | null
  text: string
  length: number
}

export interface IndexVectorsResponse {
  index: string
  total: number
  returned: number
  capped: boolean
  buckets: string[]
  points: IndexVectorPoint[]
}

export interface OnboardingStatus {
  configured: boolean
  /** true only when no LLM provider has been chosen yet — gates /welcome, not F5 after settings edits */
  needs_wizard: boolean
  llm_provider: string
  embedding_provider: string
  vector_store: string
  compute: string
}

export interface OnboardingTestRequest {
  target: 'llm' | 'embedding' | 'reranker'
  changes: Record<string, unknown>
}

export interface OnboardingTestResponse {
  ok: boolean
  detail: string
}

export interface EvalParams {
  dataset?: string | null
  cases?: Record<string, unknown>[] | null
  index_name?: string | null
  top_k?: number | null
  rerank_top_n?: number | null
  similarity_threshold?: number | null
  use_rerank?: boolean
  per_leg?: boolean
  save?: boolean
  label?: string | null
}

export interface CIRange {
  0: number
  1: number
  length: 2
}

export interface EvalCutoffMetric {
  [k: string]: number
}

export interface EvalOverall {
  hit_rate: number
  recall: number
  ndcg: number
  map: number
  mrr: number
  cases: number
  latency_ms: { p50: number; p95: number }
  ci: Record<string, [number, number]>
  recall_at: EvalCutoffMetric
  precision_at: EvalCutoffMetric
  ndcg_at: EvalCutoffMetric
}

export interface EvalCaseNode {
  rank: number
  file_name: string
  score: number | null
  text: string
  matched: boolean
}

export interface EvalExpectedChunk {
  file_name: string
  contains?: string
}

export interface EvalCase {
  query: string
  bucket?: string
  matched: number
  expected: number
  first_rank?: number
  recall: number
  precision_at: EvalCutoffMetric
  reciprocal_rank: number
  ndcg: number
  average_precision: number
  latency_ms: number
  expected_chunks?: EvalExpectedChunk[]
  retrieved_nodes?: EvalCaseNode[]
  retrieved?: number
}

export interface EvalSystemIndexMeta {
  embedding_provider?: string
  embedding_model?: string
  chunking?: ChunkingConfig
}

export interface EvalSystem {
  vector_store?: string
  indices?: Record<string, EvalSystemIndexMeta>
}

export interface EvalParamsEcho {
  system?: EvalSystem
  [key: string]: unknown
}

export interface EvalReport {
  overall: EvalOverall
  per_bucket: Record<string, EvalOverall>
  cases: EvalCase[]
  run_id?: string
  params?: EvalParamsEcho
}

export interface EvalLegOverall {
  hit_rate: number
  recall: number
  mrr: number
  ndcg: number
  map: number
}

export interface EvalLegReport {
  per_leg: Record<'bm25' | 'dense' | 'hybrid' | 'reranked', { overall: EvalLegOverall }>
  rerank_lift: Record<string, number>
}

export interface EvalRunSummary {
  id: string
  timestamp?: string
  label?: string
  overall?: Partial<EvalOverall>
  params?: EvalParamsEcho
}

export interface EvalRunsResponse {
  runs: EvalRunSummary[]
}

export interface CompareMetricDelta {
  delta: number
  ci: [number, number]
  significant: boolean
}

export interface CompareResponse {
  n_paired: number
  metrics: Record<string, CompareMetricDelta>
  a_only?: string[]
  b_only?: string[]
}
