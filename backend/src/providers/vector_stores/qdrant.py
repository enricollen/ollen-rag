"""Qdrant vector store backend: dense retrieval, index admin and metadata, self-registered with
the vector-store factory on import.

Qdrant has no native BM25 in this wiring, so this backend declares DENSE only; the retrieval layer
degrades any HYBRID/SPARSE request to DENSE via pick_supported_mode. Collections are our indices.
Build config (embedding + chunking) is stored on a reserved meta point (Qdrant has no collection-
level metadata dict like Chroma), filtered out of browse/count views.
"""
import json
import uuid
from typing import Any

import httpx
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from src.exceptions import VectorStoreError
from src.factories.vector_store import QueryMode, VectorStoreBackend, VectorStoreFactory
from src.logger import OllenLogger
from src.settings import Settings, get_settings

log = OllenLogger("qdrant_backend")

# llama-index bookkeeping keys stored alongside user metadata; hidden from browse/aggregation views.
_INTERNAL_META_KEYS = {"_node_content", "_node_type", "doc_id", "document_id", "ref_doc_id", "_ollen_index_meta"}

# reserved point holding build config for a collection (embedding provider/model + chunking).
_META_POINT_ID = str(uuid.UUID("00000000-0000-4000-8000-000000000001"))
_META_FLAG = "_ollen_index_meta"

def _build_metadata_filters(raw_filters: list[dict] | None, condition: str) -> MetadataFilters | None:
    """convert api-level filter dicts ({key, value, operator}) into llamaindex MetadataFilters."""
    if not raw_filters:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key=f["key"], value=f["value"], operator=FilterOperator(f.get("operator", "==")))
            for f in raw_filters
        ],
        condition=FilterCondition(condition),
    )

def _payload_text(payload: dict) -> str:
    """prefer an explicit text field; else recover text from llama-index's _node_content blob."""
    if payload.get("text"):
        return str(payload["text"])
    raw = payload.get("_node_content")
    if not raw:
        return ""
    try:
        return str(json.loads(raw).get("text") or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

def _public_meta(payload: dict | None) -> dict:
    """strip llama-index / ollen bookkeeping keys from a point payload."""
    return {k: v for k, v in (payload or {}).items() if k not in _INTERNAL_META_KEYS}

def _not_meta_filter() -> models.Filter:
    """exclude the reserved index-meta point from browse/count/dedup queries."""
    return models.Filter(
        must_not=[models.FieldCondition(key=_META_FLAG, match=models.MatchValue(value=True))]
    )

def _combine_filters(*parts: models.Filter | None) -> models.Filter | None:
    """merge filter clauses; empty input yields None."""
    must: list[Any] = []
    must_not: list[Any] = []
    should: list[Any] = []
    for part in parts:
        if part is None:
            continue
        must.extend(part.must or [])
        must_not.extend(part.must_not or [])
        should.extend(part.should or [])
    if not (must or must_not or should):
        return None
    return models.Filter(must=must or None, must_not=must_not or None, should=should or None)

def qdrant_reachable(settings: Settings, timeout: float = 2.0) -> bool:
    """best-effort liveness check against the configured qdrant url. any http response means
    something is listening; a connection failure means it's down.

    used by the console when the operator picks qdrant: the store is opt-in behind compose's
    `qdrant` profile, so it may simply not be running yet.
    """
    if settings.qdrant_path:
        # embedded local path -- "reachable" if the client can open it.
        try:
            QdrantClient(path=settings.qdrant_path).get_collections()
            return True
        except Exception:
            return False
    headers = {}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    try:
        r = httpx.get(f"{settings.qdrant_url.rstrip('/')}/readyz", headers=headers, timeout=timeout)
        return r.status_code < 500
    except httpx.HTTPError:
        return False

@VectorStoreFactory.register("qdrant")
class QdrantBackend(VectorStoreBackend):
    """VectorStoreBackend backed by Qdrant (server url or local path), dense retrieval only."""

    def __init__(self, settings: Settings | None = None, client: QdrantClient | None = None) -> None:
        self._settings = settings or get_settings()
        # client is the test seam (inject :memory:); else build from settings url/path.
        self._client = client or self._build_client()

    # --- capability ---
    @property
    def supported_query_modes(self) -> set[QueryMode]:
        """qdrant serves dense (kNN) retrieval only in this wiring; no bm25/hybrid yet."""
        return {QueryMode.DENSE}

    # --- internals ---
    def _build_client(self) -> QdrantClient:
        """build a qdrant client from settings: local path wins, else url (+ optional api key)."""
        s = self._settings
        try:
            if s.qdrant_path:
                return QdrantClient(path=s.qdrant_path)
            kwargs: dict[str, Any] = {"url": s.qdrant_url}
            if s.qdrant_api_key:
                kwargs["api_key"] = s.qdrant_api_key
            return QdrantClient(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Failed to connect to Qdrant: {exc}") from exc

    def _exists(self, index: str) -> bool:
        """true when the collection is present."""
        try:
            return bool(self._client.collection_exists(index))
        except Exception as exc:
            raise VectorStoreError(f"Failed to check Qdrant collection '{index}': {exc}") from exc

    def _require(self, index: str) -> None:
        """raise if the collection is missing."""
        if not self._exists(index):
            raise VectorStoreError(f"Qdrant collection '{index}' not found")

    def _vector_store(self, index: str, dim: int | None = None) -> QdrantVectorStore:
        """llama-index adapter for add/query; dense_config only needed when creating."""
        kwargs: dict[str, Any] = {"collection_name": index, "client": self._client}
        if dim is not None:
            kwargs["dense_config"] = models.VectorParams(size=int(dim), distance=models.Distance.COSINE)
        return QdrantVectorStore(**kwargs)

    def _scroll_all(
        self,
        index: str,
        *,
        scroll_filter: models.Filter | None = None,
        with_vectors: bool = False,
        limit: int | None = None,
    ) -> list[models.Record]:
        """paginate scroll until exhausted (or until *limit* records collected)."""
        out: list[models.Record] = []
        offset = None
        page = 256
        while True:
            take = page if limit is None else min(page, limit - len(out))
            if take <= 0:
                break
            try:
                records, offset = self._client.scroll(
                    collection_name=index,
                    scroll_filter=scroll_filter,
                    limit=take,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to scroll Qdrant collection '{index}': {exc}") from exc
            out.extend(records)
            if offset is None or (limit is not None and len(out) >= limit):
                break
        return out[:limit] if limit is not None else out

    # --- lifecycle ---
    def warmup(self) -> None:
        """no shared server-side pipeline to prime for qdrant."""

    def ensure_ready(self, index: str, dim: int) -> None:
        """idempotently create the collection with cosine vectors of the given dim."""
        try:
            if self._client.collection_exists(index):
                return
            self._client.create_collection(
                collection_name=index,
                vectors_config=models.VectorParams(size=int(dim), distance=models.Distance.COSINE),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to ensure Qdrant collection '{index}': {exc}") from exc

    # --- ingest ---
    def add_nodes(self, index: str, nodes: list[BaseNode]) -> None:
        """write already-embedded nodes into the collection."""
        if not nodes:
            return
        dim = len(nodes[0].embedding or [])
        if not dim:
            raise VectorStoreError(f"Cannot write nodes to Qdrant collection '{index}': missing embeddings")
        try:
            self.ensure_ready(index, dim)
            self._vector_store(index, dim).add(nodes)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to write nodes to Qdrant collection '{index}': {exc}") from exc

    # --- retrieve ---
    def retrieve(self, index, query_str, query_embedding, mode, top_k, raw_filters, filter_condition):
        """dense knn query; return raw scored nodes (threshold + rerank done by caller)."""
        self._require(index)
        store = self._vector_store(index)
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_str,
            mode=VectorStoreQueryMode.DEFAULT,
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
        """return the collection's vector dimension from its config, or None if missing."""
        if not self._exists(index):
            return None
        try:
            info = self._client.get_collection(index)
            vectors = info.config.params.vectors
            if isinstance(vectors, models.VectorParams):
                return int(vectors.size)
            # named vectors map — take the first dense config if present.
            if isinstance(vectors, dict) and vectors:
                first = next(iter(vectors.values()))
                if isinstance(first, models.VectorParams):
                    return int(first.size)
            return None
        except Exception as exc:
            raise VectorStoreError(f"Failed to read dim for Qdrant collection '{index}': {exc}") from exc

    def set_index_meta(self, index: str, embedding_provider: str, embedding_model: str, chunking: dict) -> None:
        """record build config on the reserved meta point (creates collection if needed)."""
        dim = self.get_index_dim(index)
        if dim is None:
            # collection may not exist yet — create with a placeholder dim; ensure_ready at ingest
            # will no-op once created. prefer waiting for a real dim from an existing collection.
            raise VectorStoreError(f"Qdrant collection '{index}' not found; ingest before recording meta")
        payload = {
            _META_FLAG: True,
            "embedding_provider": embedding_provider or "",
            "embedding_model": embedding_model or "",
            "chunking": json.dumps(chunking or {}),
        }
        try:
            self._client.upsert(
                collection_name=index,
                points=[
                    models.PointStruct(
                        id=_META_POINT_ID,
                        vector=[0.0] * dim,
                        payload=payload,
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to record metadata for Qdrant collection '{index}': {exc}") from exc

    def get_index_meta(self, index: str) -> dict | None:
        """return {embedding_provider, embedding_model, chunking:{...}} or None if unset/missing."""
        if not self._exists(index):
            return None
        try:
            points = self._client.retrieve(collection_name=index, ids=[_META_POINT_ID], with_payload=True)
        except Exception as exc:
            raise VectorStoreError(f"Failed to read metadata for Qdrant collection '{index}': {exc}") from exc
        if not points:
            return None
        md = points[0].payload or {}
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
        """every collection with document counts (meta point excluded), ui-shaped keys."""
        try:
            out = []
            for col in self._client.get_collections().collections:
                name = col.name
                count = self._client.count(collection_name=name, count_filter=_not_meta_filter(), exact=True).count
                out.append({"index": name, "docs.count": str(count)})
            return out
        except Exception as exc:
            raise VectorStoreError(f"Failed to list Qdrant collections: {exc}") from exc

    def get_index_documents(
        self,
        index: str,
        offset: int = 0,
        limit: int = 20,
        bucket: str | None = None,
        unbucketed: bool = False,
        file_name: str | None = None,
    ) -> dict:
        """paginate stored documents for browsing; embeddings and bookkeeping keys excluded."""
        self._require(index)
        parts: list[models.Filter | None] = [_not_meta_filter()]
        if unbucketed:
            parts.append(models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="bucket"))]))
        elif bucket:
            parts.append(models.Filter(must=[models.FieldCondition(key="bucket", match=models.MatchValue(value=bucket))]))
        if file_name:
            parts.append(
                models.Filter(must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name))])
            )
        filt = _combine_filters(*parts)
        total = self._client.count(collection_name=index, count_filter=filt, exact=True).count
        # scroll has no numeric offset; skip in python for the requested page.
        records = self._scroll_all(index, scroll_filter=filt, limit=offset + limit)
        page = records[offset:offset + limit]
        documents = [
            {
                "id": str(rec.id),
                "content": _payload_text(rec.payload or {}),
                "metadata": _public_meta(rec.payload),
            }
            for rec in page
        ]
        return {"total": total, "documents": documents}

    def get_index_vectors(self, index: str, limit: int = 2000) -> list[dict]:
        """up to *limit* chunks with embedding, text, and metadata for the indices visualizer."""
        if not self._exists(index):
            return []
        records = self._scroll_all(index, scroll_filter=_not_meta_filter(), with_vectors=True, limit=limit)
        out = []
        for rec in records:
            vec = rec.vector
            if isinstance(vec, dict):
                # named vectors — take the first dense list.
                vec = next((v for v in vec.values() if isinstance(v, list)), [])
            out.append({
                "id": str(rec.id),
                "embedding": list(vec) if vec is not None else [],
                "text": _payload_text(rec.payload or {}),
                "metadata": _public_meta(rec.payload),
            })
        return out

    def _doc_payloads(self, index: str) -> list[dict]:
        """all non-meta payloads in a collection (for bucket/file aggregation)."""
        return [(r.payload or {}) for r in self._scroll_all(index, scroll_filter=_not_meta_filter())]

    def list_buckets(self, index: str) -> list[str]:
        """distinct bucket metadata values present in a collection."""
        self._require(index)
        buckets = {md.get("bucket") for md in self._doc_payloads(index) if md.get("bucket")}
        return sorted(buckets)

    def list_bucket_files(self, index: str) -> dict[str, list[str]]:
        """map each bucket to the distinct file_names it contains."""
        self._require(index)
        out: dict[str, set] = {}
        for md in self._doc_payloads(index):
            bucket, file_name = md.get("bucket"), md.get("file_name")
            if bucket and file_name:
                out.setdefault(bucket, set()).add(file_name)
        return {b: sorted(files) for b, files in out.items()}

    def list_unbucketed_files(self, index: str) -> list[str]:
        """distinct file_names of documents stored with no bucket."""
        self._require(index)
        files = {
            md.get("file_name")
            for md in self._doc_payloads(index)
            if not md.get("bucket") and md.get("file_name")
        }
        return sorted(files)

    def find_duplicate_file(self, index: str, file_hash: str, bucket: str | None) -> str | None:
        """file_name of an already-indexed doc with the same hash+bucket, or None."""
        if not self._exists(index):
            return None
        must = [models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash))]
        if bucket is None:
            filt = _combine_filters(
                _not_meta_filter(),
                models.Filter(
                    must=must + [models.IsEmptyCondition(is_empty=models.PayloadField(key="bucket"))]
                ),
            )
        else:
            filt = _combine_filters(
                _not_meta_filter(),
                models.Filter(
                    must=must + [models.FieldCondition(key="bucket", match=models.MatchValue(value=bucket))]
                ),
            )
        try:
            records, _ = self._client.scroll(
                collection_name=index,
                scroll_filter=filt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"Duplicate check failed for '{index}': {exc}") from exc
        if not records:
            return None
        duplicate_of = (records[0].payload or {}).get("file_name", "")
        log.debug("dedup hit in %s: %s", index, duplicate_of)
        return duplicate_of

    def delete_index(self, index: str) -> None:
        """permanently delete a collection and all its documents. irreversible."""
        try:
            if not self._client.collection_exists(index):
                return
            self._client.delete_collection(index)
            log.info("collection deleted: %s", index)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete Qdrant collection '{index}': {exc}") from exc

    def delete_bucket(self, index: str, bucket: str) -> int:
        """delete every document whose metadata.bucket == bucket; return count. idempotent."""
        if not self._exists(index):
            return 0
        filt = _combine_filters(
            _not_meta_filter(),
            models.Filter(must=[models.FieldCondition(key="bucket", match=models.MatchValue(value=bucket))]),
        )
        try:
            n = self._client.count(collection_name=index, count_filter=filt, exact=True).count
            if n:
                self._client.delete(
                    collection_name=index,
                    points_selector=models.FilterSelector(filter=filt),
                )
            log.info("bucket deleted: %s/%s (%d docs)", index, bucket, n)
            return n
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete bucket '{bucket}' from '{index}': {exc}") from exc
