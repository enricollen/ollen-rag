# ollen-rag-service

A generic RAG (Retrieval-Augmented Generation) microservice. It ingests documents (PDF, Office, images) into a pluggable vector store (OpenSearch or Chroma), retrieves relevant chunks with hybrid search, and generates cited answers with an LLM. Capabilities are exposed both as a REST API and as MCP tools, so it can serve human-facing apps and AI agents alike.

## Architecture

The service is a FastAPI application (`app.py`) with a FastMCP server mounted at `/mcp`. It is organized around the three classic RAG phases:

1. **Ingestion** (`src/rag/ingestion.py`) — documents are parsed with `liteparse` (LibreOffice/ImageMagick handle Office and image formats), split into chunks using a configurable chunking strategy, embedded (watsonx.ai, local fastembed, or any LiteLLM embedding vendor), and stored in the configured vector store (OpenSearch or Chroma). Each chunking strategy writes to its own index (`{prefix}_{strategy}`). REST ingestion runs as an async background job.
2. **Retrieval** (`src/rag/retrieval.py`) — hybrid search (BM25 + dense vectors, fused by an OpenSearch search pipeline) with optional metadata filters, followed by reranking (a local cross-encoder, or any LiteLLM rerank endpoint).

   Reranked node scores are **0–1 relevance probabilities** for every provider, normalized inside the connector. Vendors disagree on the scale they return: Cohere and Jina emit a probability, while the local cross-encoder and watsonx's rerank endpoint emit an unbounded logit (a live watsonx call scored two passages at `6.902` and `-0.0005`). Each connector declares which it gets and applies a sigmoid only when needed, so nothing downstream has to know or rescale.

   This changed in the LiteLLM reranker release — previously `/api/v1/retrieve` and the MCP `retrieve` tool returned raw logits while only `/api/v1/query` sources were normalized. Ranking is unaffected (sigmoid is monotonic); only the reported numbers changed.
3. **Generation** (`src/rag/generation.py`) — the reranked chunks are passed to the LLM (watsonx.ai, or any LiteLLM vendor) with a YAML prompt template (`config/prompts/`); the answer includes numbered `[n]` citations that map to the returned `sources[]`.

All three phases pick their provider independently, so a common setup is watsonx generation with local Ollama embeddings and a local cross-encoder reranker.

Supporting modules: `src/settings.py` (env-driven configuration), `src/factories/` (provider-agnostic factories: embeddings, LLM, reranker, chunkers, and the vector-store abstraction), `src/providers/` (concrete implementations), `src/api/routes.py` (REST), `src/mcp_server.py` (MCP tools).

Everything pluggable follows the same decorator-registry pattern: a factory in `src/factories/` defines the interface, concrete providers in `src/providers/` self-register on import, and each `create_*` lazy-imports `src.providers` for the registration side effect. Providers are grouped **by capability**:

- `src/providers/llm/` — `LLMConnectorFactory`; `ConnectorLLM` adapts the configured connector to the llama_index interface `CitationQueryEngine` needs (watsonx native, plus `litellm`, `litellm-watsonx`, `litellm-ollama` via LiteLLM).
- `src/providers/embeddings/` — `EmbeddingFactory` (watsonx native, fastembed local, plus `litellm`, `litellm-watsonx`, `litellm-ollama` via LiteLLM).
- `src/providers/reranker/` — `RerankerFactory`; `ConnectorRerank` adapts the configured connector to the llama_index postprocessor interface (`sentence-transformers` local cross-encoder, plus `litellm`, `litellm-watsonx`). Every connector returns 0–1 relevance probabilities, so providers are swappable without rescaling anything downstream.
- `src/factories/model_catalog.py` — everything about which model belongs to which provider: the curated yaml catalogs, and the provider→`Settings`-field resolution that `EmbeddingFactory` and `RerankerFactory` delegate here. Each factory stays self-contained (same shape as `LLMConnectorFactory`); only the resolution logic is shared, because that is the part whose hand-written copies used to drift.
- `src/providers/vector_stores/` — `VectorStoreFactory`; a **store-agnostic** `VectorStoreBackend` interface (`src/factories/vector_store.py`) that owns retrieval, ingest writes, and all index/bucket admin, hiding llama_index behind each backend. The interface is the parity contract: every method is `@abstractmethod`, so a new backend cannot instantiate until it implements the whole surface — retrieval, `list_indices`/`get_index_documents`, bucket listing (`list_buckets`/`list_bucket_files`), dedup (`find_duplicate_file`), and lifecycle (`delete_index`, `delete_bucket`). Feature parity is enforced by the compiler, not by convention. Backends declare their `supported_query_modes` (dense/sparse/hybrid); an unsupported requested mode gracefully falls back to the richest supported one (this is the one place a new store legitimately does *less* — e.g. a dense-only store degrades hybrid to dense). Two backends ship today: **OpenSearch** (dense + sparse + hybrid) and **Chroma** (embedded, dense-only, on-disk at `OLLEN_RAG_CHROMA_PATH`); Qdrant/others are add-a-file recipes.

Adding a provider = one file in the matching `src/providers/<capability>/` folder with a `@register("name", model_field="...")` decorator + one import line in that folder's `__init__.py`, then select it via the relevant `OLLEN_RAG_*_PROVIDER` / `OLLEN_RAG_VECTOR_STORE` env var. Embedding and reranker providers also need an entry in `config/{embedding,reranker}_models.yaml` — an empty list there means "any model string", which is how the generic `litellm` provider reaches a new vendor with no code change.

## Setup

```bash
# Create a virtualenv (Python 3.12) and install dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure credentials: copy the template and fill in watsonx.ai values
cp .env.example .env
```

## Run

Local development (needs a reachable OpenSearch, e.g. from the compose stack):

```bash
uvicorn app:app --reload
```

Full local stack (service + OpenSearch + OpenSearch Dashboards):

```bash
docker compose up
```

Ports: service on `8000`, OpenSearch on `9200`, Dashboards on `5601`. The first image build is slow because of the LibreOffice layer.

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness/readiness probe |
| GET | `/api/v1/strategies` | List available chunking strategies |
| POST | `/api/v1/ingest` | Upload a document, start an async ingestion job (returns `job_id`, HTTP 202) |
| GET | `/api/v1/ingest/{job_id}` | Poll ingestion job status/result |
| POST | `/api/v1/retrieve` | Hybrid retrieval + rerank; returns scored chunks plus the separate BM25/dense/hybrid legs |
| POST | `/api/v1/query` | Full RAG: retrieval + rerank + cited LLM answer |
| POST | `/api/v1/eval/retrieval` | Score a golden dataset against retrieval (hit-rate/recall@k/MRR) |
| GET | `/api/v1/config` | Effective non-secret settings (for the UI Settings page) |
| GET | `/api/v1/indices` | List service-owned indices with doc counts (active store) |
| GET | `/api/v1/indices/overview` | Every index across **all** registered vector stores, each tagged active/inactive with its build config + bucket→files map |
| GET | `/api/v1/indices/{name}/documents` | Paginate raw stored chunks (content + metadata), optionally scoped to a `bucket` |
| GET | `/api/v1/indices/{name}/info` | Full recorded build config: embedding, chunking, dim, doc count, buckets, bucket→files map |
| GET | `/api/v1/indices/{name}/embedding` | Embedding provider/model the index was built with |
| GET | `/api/v1/indices/{name}/buckets` | Distinct `bucket` values in an index |
| DELETE | `/api/v1/indices/{name}` | Permanently delete an index and all its documents |
| DELETE | `/api/v1/indices/{name}/buckets/{bucket}` | Permanently delete one bucket (all docs where `metadata.bucket == bucket`); returns the deleted count. Idempotent |

### Ingest a document (multipart, with optional metadata JSON)

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@./mydoc.pdf" \
  -F "strategy=sentence" \
  -F 'metadata={"project": "acme", "lang": "it"}'
# -> {"job_id": "…", "status": "pending"}

# Poll the job
curl http://localhost:8000/api/v1/ingest/<job_id>
```

Job polling also returns live `progress` (0–100) and `stage`
(`parsing → chunking → enriching → storing → done`), fine-grained during the slow
LLM loops (llm chunking, keyword enrichment); the UI shows it as a progress bar on
the job card.

The KB panel accepts multiple files per submit (shared bucket/strategy/metadata/
enrichment) and processes them strictly one at a time: each file is uploaded only
after the previous ingestion finished, so heavy LLM ingests never run concurrently.
A failed file is reported and the batch continues with the next one.

Duplicate uploads are detected by content hash: every chunk stores a `file_hash`
(sha256 of the uploaded bytes) in its metadata, and ingestion is skipped when the same
hash already exists in the target index **within the same bucket** (`metadata.bucket`;
the same file in a different bucket is not a duplicate). A skipped job completes with
`"skipped_duplicate": true` and `"duplicate_of": "<existing file_name>"` in its result.

### Managing indices and buckets

The **Indices** page shows every index across all registered vector stores
(`/api/v1/indices/overview`); indices in the active store are browsable and
deletable, inactive-store ones are read-only (browse/delete target the active
backend). Two irreversible, confirmation-gated **Delete** actions are available:

- **Delete an index** — 🗑 Delete on an index card removes the whole index and all
  its documents (`DELETE /api/v1/indices/{name}`).
- **Delete a bucket** — 🗑 Delete on a 📦 bucket card removes just that bucket's
  documents (`DELETE /api/v1/indices/{name}/buckets/{bucket}`), leaving the rest of
  the index intact. The response reports how many documents were deleted; the call
  is idempotent (a missing index or bucket deletes nothing and returns `0`). Works
  identically on OpenSearch (a `_delete_by_query` on `metadata.bucket`) and Chroma
  (a `collection.delete(where={"bucket": …})`).

### LLM keyword enrichment (opt-in)

Set `enrich_keywords=true` on the ingest form (or the `OLLEN_RAG_ENRICH_KEYWORDS=true`
service default, or the MCP tool parameter) to run one LLM call per chunk at ingest
time, extracting 5–10 search keywords stored in `metadata.keywords`. Keywords are
embedded with the chunk (better dense recall) and searched by the BM25 leg via
`multi_match` on `content` + `metadata.keywords^2` (better lexical recall). Slower
ingestion; an LLM failure fails the job (re-upload retries cleanly). Already-indexed
chunks are not backfilled — re-ingest to enrich. Measure the impact with the eval
harness (`POST /api/v1/eval/retrieval`) on the same dataset before/after enabling it.

### Retrieve chunks (with metadata filters)

```bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "strategy": "sentence",
    "top_k": 10,
    "rerank_top_n": 4,
    "filters": [{"key": "project", "value": "acme", "operator": "=="}],
    "filter_condition": "and"
  }'
```

### Ask a question (cited RAG answer)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?", "strategy": "sentence"}'
# -> {"answer": "… [1] …", "sources": [{"id": 1, "text": "…", "metadata": {…}}, …]}
```

## Retrieval evaluation

Golden datasets live in `config/eval/*.yaml` (see `config/eval/example.yaml`; every case is
scoped to a mandatory `bucket`). Metrics: hit-rate@k, recall@k, MRR — overall and per bucket.

- CLI: `python -m src.rag.evaluation --dataset config/eval/golden.yaml [--top-k 10 --threshold 0.2 --no-rerank --reranker-provider litellm-watsonx]`
- API: `POST /api/v1/eval/retrieval` with `{"dataset": "golden"}` or inline `{"cases": [...]}` plus
  optional `top_k`, `rerank_top_n`, `similarity_threshold`, `use_rerank`.

Run it before and after changing hybrid weights, `similarity_threshold`, chunking strategy or the
reranker — retrieval tuning without a baseline is guesswork. Requires a running OpenSearch with
the corpus ingested.

## MCP server

The FastMCP server is mounted at `http://localhost:8000/mcp` (streamable HTTP transport). It exposes four tools, each delegating to the same rag layer as the REST API so their behavior stays in lockstep:

- `ingest_document` — parse/chunk/embed/store a **server-local** file path. Params mirror the REST ingest form: `strategy`, `index_name`, `metadata`, `enrich_keywords`, `embedding_provider`, `embedding_model`, `chunk_params`.
- `retrieve` — hybrid BM25+dense + rerank. Params: `strategy`, `index_name`, `top_k`, `rerank_top_n`, `filters`, `filter_condition`, `similarity_threshold`, `reranker_provider`, `reranker_model`.
- `rag_query` — cited RAG answer. Same params as `retrieve` plus `prompt_name`.
- `list_indices` — service-owned indices with doc counts.

Example MCP client configuration:

```json
{
  "mcpServers": {
    "ollen-rag": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Chunking strategies

By default each strategy stores its chunks in a dedicated index named after the strategy (e.g. `sentence`), so the same corpus can be indexed and compared under multiple strategies. Passing an explicit `index_name` (ingest form / MCP param) overrides this — useful to name an index by embedding model or corpus rather than strategy. **One index = one build config**: an index's `_meta` records its embedding provider/model and chunking config, and ingestion rejects a document whose embedding or chunking differs from what the index was built with.

| Strategy | Splitter | Index (default) | Notes |
|----------|----------|-----------------|-------|
| `sentence` | SentenceSplitter | `sentence` | Default; sentence-aware, `chunk_size`/`chunk_overlap` |
| `token` | TokenTextSplitter | `token` | Fixed token windows, `chunk_size`/`chunk_overlap` |
| `semantic` | SemanticSplitterNodeParser | `semantic` | Embedding-based topic breakpoints (`semantic_breakpoint_percentile`) |
| `window` | SentenceWindowNodeParser | `window` | One sentence per chunk plus a ±`sentence_window_size` context window |
| `llm` | LLM-driven topic splitter | `llm` | LLM groups sentences into topics (`llm_chunk_max_size`/`llm_chunk_window_size`); slowest |

## Configuration

All settings live in `src/settings.py` and are overridable via `OLLEN_RAG_*` environment variables (a local `.env` file is honored; see `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLEN_RAG_WATSONX_URL` | `https://eu-de.ml.cloud.ibm.com` | watsonx.ai endpoint |
| `OLLEN_RAG_WATSONX_APIKEY` | (empty) | watsonx.ai API key |
| `OLLEN_RAG_WATSONX_PROJECT_ID` | (empty) | watsonx.ai project id |
| `OLLEN_RAG_WATSONX_LLM_MODEL_ID` | `meta-llama/llama-3-3-70b-instruct` | LLM model id |
| `OLLEN_RAG_WATSONX_EMBEDDING_MODEL_ID` | `ibm/slate-125m-english-rtrvr` | Embedding model id |
| `OLLEN_RAG_WATSONX_MAX_NEW_TOKENS` | `800` | Max generated tokens |
| `OLLEN_RAG_WATSONX_TEMPERATURE` | `0.1` | LLM temperature |
| `OLLEN_RAG_WATSONX_REPETITION_PENALTY` | `1.15` | Penalizes repeated tokens. Too high (>1.3) causes garbled/merged-word output |
| `OLLEN_RAG_EMBEDDING_PROVIDER` | `watsonx` | `watsonx` (native SDK), `fastembed` (local), `litellm` (generic), `litellm-watsonx`, `litellm-ollama` |
| `OLLEN_RAG_LLM_PROVIDER` | `watsonx` | `watsonx` (native SDK), `litellm` (generic), `litellm-watsonx`, `litellm-ollama` |
| `OLLEN_RAG_RERANKER_PROVIDER` | `sentence-transformers` | `sentence-transformers` (local cross-encoder), `litellm` (generic), `litellm-watsonx`. Ollama exposes no rerank endpoint |
| `OLLEN_RAG_LITELLM_MODEL` | (empty) | Full LiteLLM model string for the generic LLM provider, e.g. `openai/gpt-4o` |
| `OLLEN_RAG_LITELLM_API_BASE` | (empty) | Endpoint override for the generic providers; the shared fallback for the two below |
| `OLLEN_RAG_LITELLM_API_KEY` | (empty) | API key for the generic providers; the shared fallback for the two below |
| `OLLEN_RAG_LITELLM_MAX_NEW_TOKENS` | `800` | Generation cap for `litellm` and `litellm-ollama` |
| `OLLEN_RAG_LITELLM_TEMPERATURE` | `0.1` | Sampling temperature for `litellm` and `litellm-ollama` |
| `OLLEN_RAG_LITELLM_EMBEDDING_MODEL` | (empty) | Full LiteLLM model string for the generic embedding provider, e.g. `openai/text-embedding-3-small` |
| `OLLEN_RAG_LITELLM_EMBEDDING_API_BASE` | (empty) | Embedding endpoint; falls back to `OLLEN_RAG_LITELLM_API_BASE` |
| `OLLEN_RAG_LITELLM_EMBEDDING_API_KEY` | (empty) | Embedding key; falls back to `OLLEN_RAG_LITELLM_API_KEY` |
| `OLLEN_RAG_LITELLM_RERANK_MODEL` | (empty) | Full LiteLLM model string for the generic rerank provider, e.g. `cohere/rerank-v3.5` |
| `OLLEN_RAG_LITELLM_RERANK_API_BASE` | (empty) | Rerank endpoint; falls back to `OLLEN_RAG_LITELLM_API_BASE` |
| `OLLEN_RAG_LITELLM_RERANK_API_KEY` | (empty) | Rerank key; falls back to `OLLEN_RAG_LITELLM_API_KEY` |
| `OLLEN_RAG_OLLAMA_API_BASE` | `http://localhost:11434` | Local Ollama endpoint |
| `OLLEN_RAG_OLLAMA_MODEL` | `llama3.1` | Bare Ollama chat model tag; the connector adds the `ollama/` prefix |
| `OLLEN_RAG_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Bare Ollama embedding model tag; the chat model cannot embed |
| `OLLEN_RAG_FASTEMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | fastembed model (local embeddings) |
| `OLLEN_RAG_VECTOR_STORE` | `opensearch` | Vector-store backend: `opensearch` (dense+sparse+hybrid) or `chroma` (embedded, dense-only). Process-global — one running service = one store, no cross-DB mixing. |
| `OLLEN_RAG_CHROMA_PATH` | `./chroma_db` | On-disk location for the embedded Chroma store (when `OLLEN_RAG_VECTOR_STORE=chroma`) |
| `OLLEN_RAG_OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch URL |
| `OLLEN_RAG_OPENSEARCH_USER` | (empty) | OpenSearch basic-auth user |
| `OLLEN_RAG_OPENSEARCH_PASSWORD` | (empty) | OpenSearch basic-auth password |
| `OLLEN_RAG_OPENSEARCH_VERIFY_CERTS` | `true` | Verify TLS certificates |
| `OLLEN_RAG_OPENSEARCH_HYBRID_PIPELINE` | `ollen-rag-hybrid` | Hybrid search pipeline name |
| `OLLEN_RAG_HYBRID_SPARSE_WEIGHT` | `0.3` | BM25 weight in hybrid fusion |
| `OLLEN_RAG_HYBRID_DENSE_WEIGHT` | `0.7` | Dense weight in hybrid fusion |
| `OLLEN_RAG_DEFAULT_CHUNKING_STRATEGY` | `sentence` | `sentence` \| `token` \| `semantic` \| `window` \| `llm` |
| `OLLEN_RAG_CHUNK_SIZE` | `512` | Chunk size (sentence/token) |
| `OLLEN_RAG_CHUNK_OVERLAP` | `64` | Chunk overlap (sentence/token) |
| `OLLEN_RAG_SEMANTIC_BREAKPOINT_PERCENTILE` | `95` | Semantic split threshold |
| `OLLEN_RAG_SENTENCE_WINDOW_SIZE` | `3` | Window size for `window` strategy |
| `OLLEN_RAG_RETRIEVAL_TOP_K` | `10` | Candidates fetched from OpenSearch |
| `OLLEN_RAG_RERANK_TOP_N` | `4` | Chunks kept after reranking |
| `OLLEN_RAG_RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Local cross-encoder, used when `OLLEN_RAG_RERANKER_PROVIDER=sentence-transformers` |
| `OLLEN_RAG_WATSONX_RERANKER_MODEL_ID` | `cross-encoder/ms-marco-minilm-l-12-v2` | Bare model id for `litellm-watsonx` rerank; the `watsonx/` prefix is added at call time |
| `OLLEN_RAG_CITATION_CHUNK_SIZE` | `512` | Citation chunk size for generation |
| `OLLEN_RAG_PROMPTS_DIR` | `config/prompts` | Prompt templates directory |
| `OLLEN_RAG_DEFAULT_PROMPT_NAME` | `rag_answer` | Default prompt template |
| `OLLEN_RAG_LOG_LEVEL` | `INFO` | `DEBUG` for per-chunk detail (keywords, skipped fragments, threshold cuts) |

## Tests

```bash
# Unit tests (integration tests are excluded by default via pytest.ini)
.venv/bin/python -m pytest

# Integration tests (require a running OpenSearch, e.g. docker compose up)
.venv/bin/python -m pytest -m integration
```

## TODO

- Add [Docling](https://github.com/docling-project/docling) as an additional parser option.
- Broaden file-type support (`.txt`, `.pptx`, `.docx`, etc.) beyond what the current parser covers.
- Add Qdrant as a vector store backend.