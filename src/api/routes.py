"""REST routes: health, ingestion (async jobs), retrieval, cited generation, index listing."""
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from src.factories.chunker import CHUNKING_STRATEGIES
from src.factories.embeddings import load_embedding_model_choices
from src.factories.vector_store import create_backend, embedding_meta
from src.rag.evaluation import evaluate, load_dataset, parse_dataset
from src.rag.generation import generate
from src.rag.ingestion import JOBS, create_job, run_ingestion_job
from src.rag.retrieval import load_reranker_model_choices, retrieve, retrieve_debug
from src.settings import get_settings

router = APIRouter()

class FilterSpec(BaseModel):
    """One metadata filter clause: key/value plus llamaindex FilterOperator symbol."""
    key: str
    value: Any
    operator: str = "=="

class RetrieveRequest(BaseModel):
    """Retrieval parameters; strategy picks the '{prefix}_{strategy}' index unless index_name is set."""
    query: str
    strategy: str | None = None
    index_name: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    filters: list[FilterSpec] | None = None
    filter_condition: str = "and"
    reranker_model: str | None = None

class QueryRequest(RetrieveRequest):
    """Generation parameters: retrieval params plus optional prompt template name."""
    prompt_name: str | None = None

class EvalRequest(BaseModel):
    """Eval run parameters: exactly one of dataset (file stem under eval_dir) or inline cases."""
    dataset: str | None = None
    cases: list[dict] | None = None
    index_name: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    use_rerank: bool = True

def _raw_filters(request: RetrieveRequest) -> list[dict] | None:
    """Convert FilterSpec models into the plain dicts the rag layer consumes."""
    if not request.filters:
        return None
    return [f.model_dump() for f in request.filters]

@router.get("/health")
def health() -> dict:
    """Liveness/readiness probe endpoint."""
    return {"status": "ok"}

@router.get("/api/v1/strategies")
def strategies() -> dict:
    """List available chunking strategies (each maps to its own index)."""
    return {"strategies": list(CHUNKING_STRATEGIES)}

@router.post("/api/v1/ingest", status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    strategy: str | None = Form(default=None),
    index_name: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
    enrich_keywords: bool | None = Form(default=None),
    embedding_provider: str | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    chunk_params: str | None = Form(default=None),
) -> dict:
    """Accept a document upload and start an async ingestion job; returns the job id."""
    extra_metadata = None
    if metadata:
        try:
            extra_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"metadata must be valid JSON: {exc}") from exc
    # Optional per-index chunk knobs (JSON object of setting_field -> value); only those relevant
    # to the chosen strategy are honored downstream, the rest fall back to server defaults
    parsed_chunk_params = None
    if chunk_params:
        try:
            parsed_chunk_params = json.loads(chunk_params)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"chunk_params must be valid JSON: {exc}") from exc
    # Persist the upload to a temp file so the background task can read it after the response
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    job = create_job()
    background_tasks.add_task(
        run_ingestion_job, job.job_id, tmp_path, strategy, index_name, extra_metadata, file.filename,
        enrich_keywords, embedding_provider, embedding_model, parsed_chunk_params,
    )
    return {"job_id": job.job_id, "status": job.status}

@router.get("/api/v1/ingest/{job_id}")
def ingest_status(job_id: str) -> dict:
    """Poll the status/result of an ingestion job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"job_id": job.job_id, "status": job.status, "detail": job.detail, "result": job.result, "progress": job.progress, "stage": job.stage}

def _serialize_nodes(nodes: list, retrieval_scores: dict[str, float] | None = None) -> list[dict]:
    """Cast off numpy/torch scalar types (e.g. numpy.float32 from the cross-encoder
    reranker) so JSON serialization doesn't choke on them. When retrieval_scores
    (node_id -> fused hybrid score) is given, each entry also carries retrieval_score
    so consumers see the pre-rerank ranking signal next to the rerank score."""
    serialized = []
    for n in nodes:
        item = {"text": n.node.get_content(), "score": float(n.score) if n.score is not None else None, "metadata": n.node.metadata}
        if retrieval_scores is not None:
            item["retrieval_score"] = retrieval_scores.get(n.node.node_id)
        serialized.append(item)
    return serialized

@router.post("/api/v1/retrieve")
def retrieve_endpoint(request: RetrieveRequest) -> dict:
    """Hybrid retrieval (BM25 + dense) with optional metadata filters, reranked by cross-encoder.

    Also returns the BM25-only, dense-only and pre-rerank fused ("hybrid") hits (the hybrid
    leg is a separate single-mode query against the same index) so the UI can show what each
    retrieval leg found before fusion/rerank. Reranked nodes[] carry a retrieval_score joined
    from the hybrid leg by node id, alongside the rerank score, so consumers can compare the
    two ranking signals side by side.
    """
    debug = retrieve_debug(
        request.query,
        strategy=request.strategy,
        index_name=request.index_name,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        similarity_threshold=request.similarity_threshold,
        raw_filters=_raw_filters(request),
        filter_condition=request.filter_condition,
        reranker_model=request.reranker_model,
    )
    # Join rerank results back to their fused scores by node id (rerank overwrites node.score)
    hybrid_scores = {n.node.node_id: float(n.score) for n in debug["hybrid"] if n.score is not None}
    return {
        "nodes": _serialize_nodes(debug["reranked"], retrieval_scores=hybrid_scores),
        "hybrid_nodes": _serialize_nodes(debug["hybrid"]),
        "bm25_nodes": _serialize_nodes(debug["bm25"]),
        "dense_nodes": _serialize_nodes(debug["dense"]),
    }

@router.post("/api/v1/query")
def query_endpoint(request: QueryRequest) -> dict:
    """Full RAG: retrieval + rerank + cited LLM answer; sources[] ids match [n] citations."""
    return generate(
        request.query,
        strategy=request.strategy,
        index_name=request.index_name,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        similarity_threshold=request.similarity_threshold,
        raw_filters=_raw_filters(request),
        filter_condition=request.filter_condition,
        prompt_name=request.prompt_name,
        reranker_model=request.reranker_model,
    )

@router.post("/api/v1/eval/retrieval")
def eval_retrieval(request: EvalRequest) -> dict:
    """Score a golden dataset against the live retrieval pipeline (hit-rate/recall@k/MRR, per bucket)."""
    if (request.dataset is None) == (request.cases is None):
        raise HTTPException(status_code=422, detail="Provide exactly one of 'dataset' or 'cases'")
    try:
        if request.cases is not None:
            dataset = parse_dataset({"cases": request.cases})
        else:
            # Stem-only names prevent path traversal out of the eval directory
            if not re.fullmatch(r"[A-Za-z0-9_-]+", request.dataset):
                raise HTTPException(status_code=422, detail="dataset must be a bare file name (letters, digits, -, _)")
            path = Path(get_settings().eval_dir) / f"{request.dataset}.yaml"
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"Eval dataset '{request.dataset}' not found")
            dataset = load_dataset(path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    report = evaluate(
        dataset,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        similarity_threshold=request.similarity_threshold,
        use_rerank=request.use_rerank,
        index_name=request.index_name,
    )
    return report.to_dict()

@router.get("/api/v1/config")
def config() -> dict:
    """Expose current effective (non-secret) settings for the UI's Settings page.

    Excludes credentials/passwords; config is env-driven and read-only from
    here — changing it requires editing .env and restarting the service.
    """
    s = get_settings()
    return {
        "embedding_provider": s.embedding_provider,
        "llm_provider": s.llm_provider,
        "watsonx_llm_model_id": s.watsonx_llm_model_id,
        "watsonx_repetition_penalty": s.watsonx_repetition_penalty,
        "watsonx_embedding_model_id": s.watsonx_embedding_model_id,
        "fastembed_model_name": s.fastembed_model_name,
        "embedding_model_choices": load_embedding_model_choices(),
        "opensearch_url": s.opensearch_url,
        "opensearch_index_prefix": s.opensearch_index_prefix,
        "opensearch_hybrid_pipeline": s.opensearch_hybrid_pipeline,
        "hybrid_sparse_weight": s.hybrid_sparse_weight,
        "hybrid_dense_weight": s.hybrid_dense_weight,
        "default_chunking_strategy": s.default_chunking_strategy,
        "enrich_keywords": s.enrich_keywords,
        "log_level": s.log_level,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "semantic_breakpoint_percentile": s.semantic_breakpoint_percentile,
        "sentence_window_size": s.sentence_window_size,
        "llm_chunk_max_size": s.llm_chunk_max_size,
        "llm_chunk_window_size": s.llm_chunk_window_size,
        "retrieval_top_k": s.retrieval_top_k,
        "similarity_threshold": s.similarity_threshold,
        "rerank_top_n": s.rerank_top_n,
        "reranker_model": s.reranker_model,
        "reranker_model_choices": load_reranker_model_choices(),
        "citation_chunk_size": s.citation_chunk_size,
        "default_prompt_name": s.default_prompt_name,
    }

@router.get("/api/v1/indices")
def indices() -> dict:
    """List service-owned indices with document counts."""
    return {"indices": create_backend().list_indices()}

@router.get("/api/v1/indices/{index_name}/documents")
def index_documents(index_name: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=200),
                    bucket: str | None = Query(default=None)) -> dict:
    """Paginate raw stored documents (content + metadata) for browsing an index, optionally scoped to a bucket."""
    return create_backend().get_index_documents(index_name, offset=offset, limit=limit, bucket=bucket)

@router.get("/api/v1/indices/{index_name}/embedding")
def index_embedding(index_name: str) -> dict:
    """Which embedding provider/model built this index (recorded at ingest time); null fields for legacy indices with no recorded meta."""
    meta = embedding_meta(create_backend(), index_name)
    return {"embedding_provider": meta["embedding_provider"] if meta else None, "embedding_model": meta["embedding_model"] if meta else None}

@router.get("/api/v1/indices/{index_name}/info")
def index_info(index_name: str) -> dict:
    """Full recorded build config of an index for the KB info panel: embedding provider/model,
    chunking config (strategy + knobs), vector dim, document count, and buckets present.
    Null/empty fields for a legacy index with no recorded _meta."""
    backend = create_backend()
    meta = backend.get_index_meta(index_name) or {}
    docs_count = next((int(ix.get("docs.count") or 0) for ix in backend.list_indices() if ix.get("index") == index_name), None)
    return {
        "index": index_name,
        "embedding_provider": meta.get("embedding_provider"),
        "embedding_model": meta.get("embedding_model"),
        "chunking": meta.get("chunking"),
        "dim": backend.get_index_dim(index_name),
        "docs_count": docs_count,
        "buckets": backend.list_buckets(index_name),
        "bucket_files": backend.list_bucket_files(index_name),
    }

@router.get("/api/v1/indices/{index_name}/buckets")
def index_buckets(index_name: str) -> dict:
    """Distinct 'bucket' metadata values in an index, for preloading UI dropdowns."""
    return {"buckets": create_backend().list_buckets(index_name)}

@router.delete("/api/v1/indices/{index_name}")
def index_delete(index_name: str) -> dict:
    """Permanently delete an index and all its documents. Irreversible."""
    create_backend().delete_index(index_name)
    return {"deleted": index_name}
