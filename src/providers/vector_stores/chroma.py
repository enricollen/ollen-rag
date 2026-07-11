"""Chroma vector store backend: embedded PersistentClient, dense retrieval, index admin and
metadata, self-registered with the vector-store factory on import.

Chroma has no native BM25, so this backend declares DENSE only; the retrieval layer degrades any
HYBRID/SPARSE request to DENSE via pick_supported_mode. Chroma collections are our indices, and a
collection's flat metadata dict holds the same build config (embedding + chunking) OpenSearch keeps
in its mapping _meta, so 'one index = one build config' holds across backends.
"""
import json
import chromadb
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.exceptions import VectorStoreError
from src.factories.vector_store import QueryMode, VectorStoreBackend, VectorStoreFactory
from src.logger import OllenLogger
from src.settings import Settings, get_settings

log = OllenLogger("chroma_backend")

# llama-index bookkeeping keys stored alongside user metadata in each Chroma record; hidden from the
# browse/aggregation views so the UI (and bucket/file grouping) only ever sees real document metadata.
_INTERNAL_META_KEYS = {"_node_content", "_node_type", "doc_id", "document_id", "ref_doc_id"}

def _build_metadata_filters(raw_filters: list[dict] | None, condition: str) -> MetadataFilters | None:
    """Convert API-level filter dicts ({key, value, operator}) into llamaindex MetadataFilters;
    ChromaVectorStore translates these into a Chroma `where` clause."""
    if not raw_filters:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key=f["key"], value=f["value"], operator=FilterOperator(f.get("operator", "==")))
            for f in raw_filters
        ],
        condition=FilterCondition(condition),
    )

@VectorStoreFactory.register("chroma")
class ChromaBackend(VectorStoreBackend):
    """VectorStoreBackend backed by an embedded (PersistentClient) Chroma database."""

    def __init__(self, settings: Settings | None = None, client: chromadb.api.ClientAPI | None = None) -> None:
        self._settings = settings or get_settings()
        # client is the test seam (inject a tmp PersistentClient); else a persistent client on disk.
        self._client = client or self._persistent_client()

    # --- capability ---
    @property
    def supported_query_modes(self) -> set[QueryMode]:
        """Chroma serves dense (kNN) retrieval only; no native BM25/hybrid."""
        return {QueryMode.DENSE}

    # --- internals ---
    def _persistent_client(self) -> chromadb.api.ClientAPI:
        """Build the on-disk Chroma client rooted at the configured path."""
        try:
            return chromadb.PersistentClient(path=self._settings.chroma_path)
        except Exception as exc:  # chromadb raises assorted errors on a bad path/store
            raise VectorStoreError(f"Failed to open Chroma store at '{self._settings.chroma_path}': {exc}") from exc

    def _open(self, index: str):
        """Return the existing collection or raise VectorStoreError if it doesn't exist."""
        try:
            return self._client.get_collection(index)
        except Exception as exc:
            raise VectorStoreError(f"Chroma collection '{index}' not found: {exc}") from exc

    def _maybe(self, index: str):
        """Return the collection if it exists, else None (for meta/dim lookups on possibly-missing indices)."""
        try:
            return self._client.get_collection(index)
        except Exception:
            return None

    # --- lifecycle ---
    def warmup(self) -> None:
        """No shared server-side pipeline to prime for Chroma."""

    def ensure_ready(self, index: str, dim: int) -> None:
        """Idempotently create the collection; stamp the vector dimension at creation time."""
        try:
            self._client.get_or_create_collection(index, metadata={"dim": int(dim)})
        except Exception as exc:
            raise VectorStoreError(f"Failed to ensure Chroma collection '{index}': {exc}") from exc

    # --- ingest ---
    def add_nodes(self, index: str, nodes: list[BaseNode]) -> None:
        """Write already-embedded nodes into the collection."""
        if not nodes:
            return
        try:
            col = self._client.get_or_create_collection(index)
            ChromaVectorStore(chroma_collection=col).add(nodes)
        except Exception as exc:
            raise VectorStoreError(f"Failed to write nodes to Chroma collection '{index}': {exc}") from exc

    # --- retrieve ---
    def retrieve(self, index, query_str, query_embedding, mode, top_k, raw_filters, filter_condition):
        """Dense kNN query; return the raw scored NodeWithScore list (threshold + rerank done by caller)."""
        store = ChromaVectorStore(chroma_collection=self._open(index))
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_str,
            mode=VectorStoreQueryMode.DEFAULT,  # DENSE only; caller already degraded via pick_supported_mode
            similarity_top_k=top_k,
            filters=_build_metadata_filters(raw_filters, filter_condition),
        )
        try:
            result = store.query(query)
        except Exception as exc:
            raise VectorStoreError(f"Retrieval failed on collection '{index}': {exc}") from exc
        sims = result.similarities or [None] * len(result.nodes)
        return [NodeWithScore(node=node, score=score) for node, score in zip(result.nodes, sims)]

    # --- config / meta ---
    def get_index_dim(self, index: str) -> int | None:
        """Return the collection's vector dimension: from stamped metadata, else peeked from a stored vector."""
        col = self._maybe(index)
        if col is None:
            return None
        dim = (col.metadata or {}).get("dim")
        if dim is not None:
            return int(dim)
        # Legacy collection without a stamped dim: peek one stored embedding.
        got = col.get(limit=1, include=["embeddings"])
        embeddings = got.get("embeddings")
        return len(embeddings[0]) if embeddings is not None and len(embeddings) else None

    def set_index_meta(self, index: str, embedding_provider: str, embedding_model: str, chunking: dict) -> None:
        """Record the build config in the collection's flat metadata: embedding provider/model plus the
        chunking config JSON-encoded (Chroma metadata values must be scalars). Preserves the stamped dim."""
        col = self._client.get_or_create_collection(index)
        meta = dict(col.metadata or {})
        meta.update({
            "embedding_provider": embedding_provider or "",
            "embedding_model": embedding_model or "",
            "chunking": json.dumps(chunking or {}),
        })
        try:
            col.modify(metadata=meta)
        except Exception as exc:
            raise VectorStoreError(f"Failed to record metadata for Chroma collection '{index}': {exc}") from exc

    def get_index_meta(self, index: str) -> dict | None:
        """Return the recorded build config as {embedding_provider, embedding_model, chunking:{...}},
        matching the OpenSearch shape, or None for a missing/legacy collection with no recorded meta."""
        col = self._maybe(index)
        if col is None:
            return None
        md = col.metadata or {}
        if not md.get("embedding_provider"):
            return None
        chunking = md.get("chunking")
        return {
            "embedding_provider": md["embedding_provider"],
            "embedding_model": md.get("embedding_model", ""),
            "chunking": json.loads(chunking) if chunking else {},
        }

    # --- introspection / admin ---
    def list_indices(self) -> list[dict]:
        """Return every collection in the store with document counts, keyed to match the UI
        (`index`, `docs.count`) so both backends render through the same code path. No name
        filtering — the console shows all indices regardless of naming."""
        try:
            out = []
            for col in self._client.list_collections():
                name = col.name if hasattr(col, "name") else str(col)
                count = self._client.get_collection(name).count()
                out.append({"index": name, "docs.count": str(count)})
            return out
        except Exception as exc:
            raise VectorStoreError(f"Failed to list Chroma collections: {exc}") from exc

    def get_index_documents(self, index: str, offset: int = 0, limit: int = 20, bucket: str | None = None) -> dict:
        """Paginate stored documents (content + metadata) for browsing, optionally scoped to one bucket;
        embeddings and llama-index bookkeeping keys are excluded so the UI stays readable. Page order is
        Chroma's own (no server sort)."""
        col = self._open(index)
        where = {"bucket": bucket} if bucket else None
        # Chroma's count() ignores `where`, so a filtered total = number of ids matching the bucket.
        total = len(col.get(where=where, include=[]).get("ids") or []) if where else col.count()
        got = col.get(where=where, offset=offset, limit=limit, include=["documents", "metadatas"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        documents = [
            {
                "id": ids[i],
                "content": docs[i] if i < len(docs) else "",
                "metadata": {k: v for k, v in (metas[i] or {}).items() if k not in _INTERNAL_META_KEYS},
            }
            for i in range(len(ids))
        ]
        return {"total": total, "documents": documents}

    def _all_metadatas(self, index: str) -> list[dict]:
        """Fetch every record's metadata (Chroma has no server-side aggregation; grouped in Python)."""
        col = self._open(index)
        return col.get(include=["metadatas"]).get("metadatas") or []

    def list_buckets(self, index: str) -> list[str]:
        """Return distinct 'bucket' metadata values present in a collection."""
        buckets = {md.get("bucket") for md in self._all_metadatas(index) if md.get("bucket")}
        return sorted(buckets)

    def list_bucket_files(self, index: str) -> dict[str, list[str]]:
        """Map each bucket to the distinct document file_names it contains (bucket -> [file_name])."""
        out: dict[str, set] = {}
        for md in self._all_metadatas(index):
            bucket, file_name = md.get("bucket"), md.get("file_name")
            if bucket and file_name:
                out.setdefault(bucket, set()).add(file_name)
        return {b: sorted(files) for b, files in out.items()}

    def find_duplicate_file(self, index: str, file_hash: str, bucket: str | None) -> str | None:
        """file_name of an already-indexed doc with the same hash+bucket, or None. Dedup scope is
        hash AND bucket (bucket separation invariant); bucketless uploads dedupe only against other
        bucketless docs. Chroma `where` has no exists-operator, so the bucket match is done in Python."""
        col = self._maybe(index)
        if col is None:
            return None
        try:
            got = col.get(where={"file_hash": file_hash}, include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Duplicate check failed for '{index}': {exc}") from exc
        for md in got.get("metadatas") or []:
            if (md.get("bucket") or None) == bucket:
                duplicate_of = md.get("file_name", "")
                log.debug("dedup hit in %s: %s", index, duplicate_of)
                return duplicate_of
        return None

    def delete_index(self, index: str) -> None:
        """Permanently delete a collection and all its documents. Irreversible."""
        try:
            self._client.delete_collection(index)
            log.info("collection deleted: %s", index)
        except Exception as exc:
            # A missing collection is not an error (idempotent delete), mirroring the OpenSearch backend.
            if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
                return
            raise VectorStoreError(f"Failed to delete Chroma collection '{index}': {exc}") from exc

    def delete_bucket(self, index: str, bucket: str) -> int:
        """Delete every document in `index` whose metadata.bucket == `bucket`.
        Returns the number of documents deleted. Idempotent: a missing index or
        bucket deletes nothing and returns 0. Chroma has no server-side count,
        so we fetch the matching ids first and delete by the same `where`."""
        col = self._maybe(index)
        if col is None:
            return 0
        try:
            ids = col.get(where={"bucket": bucket}).get("ids") or []
            if ids:
                col.delete(where={"bucket": bucket})
            log.info("bucket deleted: %s/%s (%d docs)", index, bucket, len(ids))
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete bucket '{bucket}' from '{index}': {exc}") from exc
