"""Ingestion flow: liteparse parsing -> chunking -> embedding -> OpenSearch storage, plus async job tracking."""
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from liteparse import LiteParse
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from src.exceptions import ParsingError
from src.factories.chunker import CHUNKING_STRATEGIES, create_node_parser
from src.factories.embeddings import create_embedding_model, get_embedding_dim, load_embedding_model_choices
from src.factories.llm import create_llm
from src.factories.vector_store import build_index_name, create_backend
from src.logging_config import OllenLogger
from src.prompts import load_prompt
from src.rag.enrichment import KeywordEnricher
from src.settings import get_settings

log = OllenLogger("ingestion")

# Which Settings fields are the meaningful chunk knobs for each strategy — used both to build
# the per-request Settings override and to record/compare the effective config in the index _meta.
# One index = one chunking config, so only the relevant knobs are recorded per strategy.
CHUNK_PARAM_FIELDS: dict[str, tuple[str, ...]] = {
    "sentence": ("chunk_size", "chunk_overlap"),
    "token": ("chunk_size", "chunk_overlap"),
    "semantic": ("semantic_breakpoint_percentile",),
    "window": ("sentence_window_size",),
    "llm": ("llm_chunk_max_size", "llm_chunk_window_size"),
}


def effective_chunking(strategy: str, settings) -> dict:
    """The chunking config that actually built (or will build) an index: strategy + the knobs
    relevant to that strategy, read from the (possibly overridden) settings."""
    config = {"strategy": strategy}
    for field in CHUNK_PARAM_FIELDS.get(strategy, ()):
        config[field] = getattr(settings, field)
    return config


def compute_file_hash(path: str | Path) -> str:
    """sha256 over the raw file bytes — the document identity used for duplicate detection."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        # Stream in 1 MiB blocks so large uploads don't load fully into memory
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_file(path: str | Path, file_name: str | None = None) -> list[Document]:
    """Parse any supported file (PDF/Office/images) to markdown and wrap it as a llamaindex Document."""
    parser = LiteParse(output_format="markdown")
    try:
        result = parser.parse(str(path))
    except Exception as exc:
        raise ParsingError(f"Failed to parse '{file_name or path}': {exc}") from exc
    if not result.text or not result.text.strip():
        raise ParsingError(f"No text content extracted from '{file_name or path}'")
    metadata = {"file_name": file_name or Path(path).name, "num_pages": len(result.pages)}
    return [Document(text=result.text, metadata=metadata)]


def ingest_document(
    path: str | Path,
    strategy: str | None = None,
    index_name: str | None = None,
    extra_metadata: dict | None = None,
    file_name: str | None = None,
    enrich_keywords: bool | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    chunk_params: dict | None = None,
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict:
    """Full ingestion: parse -> chunk (per strategy) -> optional LLM keyword enrichment -> embed -> store.

    progress_cb, when given, receives (percent 0-100, stage) at stage boundaries and,
    for the slow LLM loops (llm chunking, keyword enrichment), per-item updates.
    """
    started = time.perf_counter()

    def report(pct: int, stage: str) -> None:
        # Single funnel for progress so a None callback costs nothing
        if progress_cb is not None:
            progress_cb(pct, stage)

    settings = get_settings()
    # Per-request embed model override: build a scoped Settings copy so the rest of the
    # pipeline (create_embedding_model, node parser, vector store) stays settings-blind
    if embedding_provider or embedding_model:
        provider = embedding_provider or settings.embedding_provider
        model_field = {"watsonx": "watsonx_embedding_model_id", "fastembed": "fastembed_model_name"}.get(provider)
        if model_field is None:
            raise ValueError(f"Unknown embedding provider '{provider}'. Available: {sorted(load_embedding_model_choices())}")
        if embedding_model and embedding_model not in load_embedding_model_choices().get(provider, []):
            raise ValueError(f"Unknown embedding model '{embedding_model}' for provider '{provider}'. Available: {load_embedding_model_choices().get(provider, [])}")
        overrides = {"embedding_provider": provider}
        if embedding_model:
            overrides[model_field] = embedding_model
        settings = settings.model_copy(update=overrides)
    strategy = strategy or settings.default_chunking_strategy
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid: {CHUNKING_STRATEGIES}")
    # Per-request chunk-param overrides: only the knobs relevant to this strategy are honored,
    # so an index's recorded config stays clean (a chunk_size on a window index is meaningless).
    if chunk_params:
        allowed = CHUNK_PARAM_FIELDS.get(strategy, ())
        overrides = {k: v for k, v in chunk_params.items() if k in allowed and v is not None}
        if overrides:
            settings = settings.model_copy(update=overrides)
    # Explicit request value wins; None falls back to the service-wide default
    enrich = settings.enrich_keywords if enrich_keywords is None else enrich_keywords
    target_index = build_index_name(strategy, index_name, settings)
    backend = create_backend(settings)
    # Duplicate detection before the expensive parse/embed work: same bytes + same bucket
    # already in the target index -> skip re-indexing (bucket separation invariant respected)
    file_hash = compute_file_hash(path)
    bucket = (extra_metadata or {}).get("bucket")
    duplicate_of = backend.find_duplicate_file(target_index, file_hash, bucket)
    if duplicate_of is not None:
        log.info("duplicate skip: '%s' already indexed as '%s' in %s", file_name or path, duplicate_of, target_index)
        return {
            "index": target_index,
            "strategy": strategy,
            "num_documents": 0,
            "num_nodes": 0,
            "file_hash": file_hash,
            "skipped_duplicate": True,
            "duplicate_of": duplicate_of,
            # Nothing was (re)indexed, so no keywords were generated regardless of the request
            "enriched": False,
        }
    log.info("ingest start: file=%s strategy=%s bucket=%s index=%s enrich=%s", file_name or path, strategy, bucket, target_index, enrich)
    report(2, "parsing")
    documents = parse_file(path, file_name)
    log.info("parsed '%s': %d page(s), %d chars", file_name or path, documents[0].metadata.get("num_pages", 0), len(documents[0].text))
    report(10, "chunking")
    # Custom metadata propagates from Documents to chunks, enabling metadata filters at retrieval;
    # file_hash is stamped on every chunk so future uploads can be dedup-checked against the index,
    # but excluded from embed/LLM text (a sha256 hex string is noise there) — exclusion lists
    # propagate from Document to chunks just like metadata does
    for doc in documents:
        if extra_metadata:
            doc.metadata.update(extra_metadata)
        doc.metadata["file_hash"] = file_hash
        doc.excluded_embed_metadata_keys.append("file_hash")
        doc.excluded_llm_metadata_keys.append("file_hash")
    # Map component-level fractions (0-1) into the stage percent bands
    chunk_band_end = 60 if enrich else 70

    def _chunk_progress(frac: float) -> None:
        pct = int(10 + frac * (chunk_band_end - 10))
        # Without enrichment the pipeline goes straight to embed+store after chunking
        report(pct, "storing" if frac >= 1.0 and not enrich else "chunking")

    def _enrich_progress(frac: float) -> None:
        report(int(60 + frac * 25), "storing" if frac >= 1.0 else "enriching")

    embed_model = create_embedding_model(settings)
    # One LLM serves both the llm chunking strategy and keyword enrichment when either needs it
    llm = create_llm(settings) if strategy == "llm" or enrich else None
    node_parser = create_node_parser(strategy, embed_model=embed_model, llm=llm, settings=settings, progress_cb=_chunk_progress)
    embed_dim = get_embedding_dim(embed_model)
    resolved_model = settings.watsonx_embedding_model_id if settings.embedding_provider == "watsonx" else settings.fastembed_model_name
    chunking = effective_chunking(strategy, settings)
    # One index = one build config (embedding model + chunking): mixing pollutes retrieval/eval.
    # Adding documents to an existing index is fine only if every recorded knob matches; otherwise
    # create a new index. Checked against the recorded _meta (the dim check alone can't catch two
    # same-dim models or a different chunk size).
    existing_meta = backend.get_index_meta(target_index)
    if existing_meta:
        if (existing_meta.get("embedding_provider"), existing_meta.get("embedding_model")) != (settings.embedding_provider, resolved_model):
            raise ValueError(
                f"index '{target_index}' was built with {existing_meta.get('embedding_provider')}/{existing_meta.get('embedding_model')}; "
                f"requested {settings.embedding_provider}/{resolved_model} — one index holds a single embedding model. "
                f"Create a new index or pick the matching model."
            )
        existing_chunking = existing_meta.get("chunking") or {}
        if existing_chunking and existing_chunking != chunking:
            raise ValueError(
                f"index '{target_index}' was built with chunking {existing_chunking}; requested {chunking} — "
                f"one index holds a single chunking config. Create a new index or match the existing config."
            )
    # Fallback for legacy indices with no recorded meta: at least block a dimension mismatch.
    existing_dim = backend.get_index_dim(target_index)
    if existing_dim is not None and existing_dim != embed_dim:
        raise ValueError(
            f"index '{target_index}' built with {existing_dim}-dim vectors; requested model "
            f"produces {embed_dim}-dim — pick a different index_name or a matching embed model"
        )
    backend.ensure_ready(target_index, embed_dim)
    # IngestionPipeline runs chunking (+ optional keyword enrichment) + embedding, producing
    # embedded nodes; the backend then writes them (no vector_store sink on the pipeline).
    transformations: list = [node_parser]
    if enrich:
        transformations.append(KeywordEnricher(llm=llm, prompt=load_prompt("keywords", settings), progress_cb=_enrich_progress))
    transformations.append(embed_model)
    pipeline = IngestionPipeline(transformations=transformations)
    nodes = pipeline.run(documents=documents)
    backend.add_nodes(target_index, nodes)
    backend.set_index_meta(target_index, settings.embedding_provider, resolved_model, chunking)
    log.info("ingest done: %d chunk(s) -> %s in %.1fs", len(nodes), target_index, time.perf_counter() - started)
    return {
        "index": target_index,
        "strategy": strategy,
        "num_documents": len(documents),
        "num_nodes": len(nodes),
        "file_hash": file_hash,
        "enriched": enrich,
        # The effective chunking config recorded on the index, so the UI can echo it back
        "chunking": chunking,
    }


@dataclass
class IngestionJob:
    """Tracks one async ingestion request through pending/running/completed/failed states."""
    job_id: str
    status: str = "pending"
    detail: str | None = None
    result: dict | None = field(default=None)
    progress: int = 0          # 0-100, monotonic
    stage: str | None = None   # parsing | chunking | enriching | storing | done


# In-memory job registry; acceptable for single-replica deployments
JOBS: dict[str, IngestionJob] = {}


def create_job() -> IngestionJob:
    """Register a new pending ingestion job and return it."""
    job = IngestionJob(job_id=str(uuid.uuid4()))
    JOBS[job.job_id] = job
    return job


def run_ingestion_job(
    job_id: str,
    path: str,
    strategy: str | None,
    index_name: str | None,
    extra_metadata: dict | None,
    file_name: str | None,
    enrich_keywords: bool | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    chunk_params: dict | None = None,
) -> None:
    """Background task body: run ingestion, record outcome, always delete the temp upload."""
    job = JOBS[job_id]
    job.status = "running"

    def _update(pct: int, stage: str) -> None:
        # Clamp + monotonic: band-math bugs must never move the bar backwards
        job.progress = max(job.progress, min(100, max(0, pct)))
        job.stage = stage

    try:
        job.result = ingest_document(
            path, strategy, index_name, extra_metadata, file_name, enrich_keywords,
            embedding_provider, embedding_model, chunk_params, progress_cb=_update,
        )
        job.status = "completed"
        job.progress = 100
        job.stage = "done"
    except Exception as exc:
        job.status = "failed"
        job.detail = str(exc)
        log.error("ingestion job %s failed: %s", job_id, exc)
    finally:
        Path(path).unlink(missing_ok=True)
