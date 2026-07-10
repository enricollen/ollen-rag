# ollen-rag-service

A generic RAG (Retrieval-Augmented Generation) microservice. It ingests documents (PDF, Office, images) into OpenSearch, retrieves relevant chunks with hybrid search, and generates cited answers with an LLM. Capabilities are exposed both as a REST API and as MCP tools, so it can serve human-facing apps and AI agents alike.

## Architecture

The service is a FastAPI application (`app.py`) with a FastMCP server mounted at `/mcp`. It is organized around the three classic RAG phases:

1. **Ingestion** (`src/rag/ingestion.py`) — documents are parsed with `liteparse` (LibreOffice/ImageMagick handle Office and image formats), split into chunks using a configurable chunking strategy, embedded (watsonx.ai or fastembed), and stored in OpenSearch. Each chunking strategy writes to its own index (`{prefix}_{strategy}`). REST ingestion runs as an async background job.
2. **Retrieval** (`src/rag/retrieval.py`) — hybrid search (BM25 + dense vectors, fused by an OpenSearch search pipeline) with optional metadata filters, followed by cross-encoder reranking.
3. **Generation** (`src/rag/generation.py`) — the reranked chunks are passed to the LLM (watsonx.ai) with a YAML prompt template (`config/prompts/`); the answer includes numbered `[n]` citations that map to the returned `sources[]`.

Supporting modules: `src/settings.py` (env-driven configuration), `src/factories/` (provider-agnostic factories: embeddings, LLM, chunkers, and the vector-store abstraction), `src/providers/` (concrete implementations), `src/api/routes.py` (REST), `src/mcp_server.py` (MCP tools).

Everything pluggable follows the same decorator-registry pattern: a factory in `src/factories/` defines the interface, concrete providers in `src/providers/` self-register on import, and each `create_*` lazy-imports `src.providers` for the registration side effect. Providers are grouped **by capability**:

- `src/providers/llm/` — `LLMConnectorFactory`; `ConnectorLLM` adapts the configured connector to the llama_index interface `CitationQueryEngine` needs (watsonx).
- `src/providers/embeddings/` — `EmbeddingFactory` (watsonx, fastembed).
- `src/providers/vector_stores/` — `VectorStoreFactory`; a **store-agnostic** `VectorStoreBackend` interface (`src/factories/vector_store.py`) that owns retrieval, ingest writes, and all index admin, hiding llama_index behind each backend. Backends declare their `supported_query_modes` (dense/sparse/hybrid); an unsupported requested mode gracefully falls back to the richest supported one. OpenSearch is the only backend today; the layer is designed so Chroma/Qdrant become add-a-file recipes.

Adding a provider = one file in the matching `src/providers/<capability>/` folder with a `@register("name")` decorator + one import line in that folder's `__init__.py`, then select it via the relevant `OLLEN_RAG_*_PROVIDER` / `OLLEN_RAG_VECTOR_STORE` env var.

## Setup

```bash
# Create a virtualenv (Python 3.13) and install dependencies
python3.13 -m venv .venv
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
| GET | `/api/v1/indices` | List service-owned indices with doc counts |
| GET | `/api/v1/indices/{name}/documents` | Paginate raw stored chunks (content + metadata) |
| GET | `/api/v1/indices/{name}/info` | Full recorded build config: embedding, chunking, dim, doc count, buckets, bucket→files map |
| GET | `/api/v1/indices/{name}/embedding` | Embedding provider/model the index was built with |
| GET | `/api/v1/indices/{name}/buckets` | Distinct `bucket` values in an index |
| DELETE | `/api/v1/indices/{name}` | Permanently delete an index and all its documents |

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

- CLI: `python -m src.rag.evaluation --dataset config/eval/golden.yaml [--top-k 10 --threshold 0.2 --no-rerank]`
- API: `POST /api/v1/eval/retrieval` with `{"dataset": "golden"}` or inline `{"cases": [...]}` plus
  optional `top_k`, `rerank_top_n`, `similarity_threshold`, `use_rerank`.

Run it before and after changing hybrid weights, `similarity_threshold`, chunking strategy or the
reranker — retrieval tuning without a baseline is guesswork. Requires a running OpenSearch with
the corpus ingested.

## MCP server

The FastMCP server is mounted at `http://localhost:8000/mcp` (streamable HTTP transport). It exposes four tools, each delegating to the same rag layer as the REST API so their behavior stays in lockstep:

- `ingest_document` — parse/chunk/embed/store a **server-local** file path. Params mirror the REST ingest form: `strategy`, `index_name`, `metadata`, `enrich_keywords`, `embedding_provider`, `embedding_model`, `chunk_params`.
- `retrieve` — hybrid BM25+dense + rerank. Params: `strategy`, `index_name`, `top_k`, `rerank_top_n`, `filters`, `filter_condition`, `similarity_threshold`, `reranker_model`.
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

By default each strategy stores its chunks in a dedicated index named `{OLLEN_RAG_OPENSEARCH_INDEX_PREFIX}_{strategy}` (default prefix `ollen_rag`), so the same corpus can be indexed and compared under multiple strategies. Passing an explicit `index_name` (ingest form / MCP param) overrides this — useful to name an index by embedding model or corpus rather than strategy. **One index = one build config**: an index's `_meta` records its embedding provider/model and chunking config, and ingestion rejects a document whose embedding or chunking differs from what the index was built with.

| Strategy | Splitter | Index (default prefix) | Notes |
|----------|----------|------------------------|-------|
| `sentence` | SentenceSplitter | `ollen_rag_sentence` | Default; sentence-aware, `chunk_size`/`chunk_overlap` |
| `token` | TokenTextSplitter | `ollen_rag_token` | Fixed token windows, `chunk_size`/`chunk_overlap` |
| `semantic` | SemanticSplitterNodeParser | `ollen_rag_semantic` | Embedding-based topic breakpoints (`semantic_breakpoint_percentile`) |
| `window` | SentenceWindowNodeParser | `ollen_rag_window` | One sentence per chunk plus a ±`sentence_window_size` context window |
| `llm` | LLM-driven topic splitter | `ollen_rag_llm` | LLM groups sentences into topics (`llm_chunk_max_size`/`llm_chunk_window_size`); slowest |

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
| `OLLEN_RAG_EMBEDDING_PROVIDER` | `watsonx` | `watsonx` or `fastembed` |
| `OLLEN_RAG_LLM_PROVIDER` | `watsonx` | LLM provider |
| `OLLEN_RAG_FASTEMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | fastembed model (local embeddings) |
| `OLLEN_RAG_VECTOR_STORE` | `opensearch` | Vector-store backend: `opensearch` (dense+sparse+hybrid) or `chroma` (embedded, dense-only). Process-global — one running service = one store, no cross-DB mixing. |
| `OLLEN_RAG_CHROMA_PATH` | `./chroma_db` | On-disk location for the embedded Chroma store (when `OLLEN_RAG_VECTOR_STORE=chroma`) |
| `OLLEN_RAG_OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch URL |
| `OLLEN_RAG_OPENSEARCH_USER` | (empty) | OpenSearch basic-auth user |
| `OLLEN_RAG_OPENSEARCH_PASSWORD` | (empty) | OpenSearch basic-auth password |
| `OLLEN_RAG_OPENSEARCH_VERIFY_CERTS` | `true` | Verify TLS certificates |
| `OLLEN_RAG_OPENSEARCH_INDEX_PREFIX` | `ollen_rag` | Index name prefix |
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
| `OLLEN_RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder rerank model |
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

