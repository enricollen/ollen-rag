"""OpenSearch vector store backend: hybrid BM25+kNN retrieval, index admin, and metadata,
self-registered with the vector-store factory on import."""
import httpx
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.opensearch import OpensearchVectorClient, OpensearchVectorStore
from src.exceptions import VectorStoreError
from src.factories.vector_store import QueryMode, VectorStoreBackend, VectorStoreFactory
from src.logger import OllenLogger
from src.settings import Settings, get_settings

log = OllenLogger("opensearch_backend")

# QueryMode -> llamaindex VectorStoreQueryMode. HYBRID rides the score-normalization search pipeline.
_MODE_MAP = {
    QueryMode.DENSE: VectorStoreQueryMode.DEFAULT,
    QueryMode.SPARSE: VectorStoreQueryMode.TEXT_SEARCH,
    QueryMode.HYBRID: VectorStoreQueryMode.HYBRID,
}

class OllenOpensearchVectorClient(OpensearchVectorClient):
    """OpensearchVectorClient with a widened BM25 leg: content + LLM-extracted keywords.

    _hybrid_search_query delegates to _lexical_search_query internally, so this single
    override covers both the BM25-only leg and the hybrid query. Applied unconditionally:
    multi_match on an unmapped field contributes no hits, so indices ingested without
    keyword enrichment behave exactly like the stock client.
    """
    def _lexical_search_query(self, text_field, query_str, k, filters=None, excluded_source_fields=None):
        """Same shape as the parent implementation, with multi_match replacing the single-field match."""
        lexical_query = {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query_str,
                        # keywords boosted ^2: a short curated field beats prose on term hits;
                        # fixed value — introduce a setting only if eval data demands tuning
                        "fields": [text_field, "metadata.keywords^2"],
                    }
                }
            }
        }
        parsed_filters = self._parse_filters(filters)
        if len(parsed_filters) > 0:
            lexical_query["bool"]["filter"] = parsed_filters
        query = {"size": k, "query": lexical_query}
        if excluded_source_fields:
            query["_source"] = {"exclude": excluded_source_fields}
        return query

def _build_metadata_filters(raw_filters: list[dict] | None, condition: str) -> MetadataFilters | None:
    """Convert API-level filter dicts ({key, value, operator}) into llamaindex MetadataFilters."""
    if not raw_filters:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key=f["key"], value=f["value"], operator=FilterOperator(f.get("operator", "==")))
            for f in raw_filters
        ],
        condition=FilterCondition(condition),
    )

@VectorStoreFactory.register("opensearch")
class OpenSearchBackend(VectorStoreBackend):
    """VectorStoreBackend backed by OpenSearch with a server-side hybrid search pipeline."""

    def __init__(self, settings: Settings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        # http_client is the test seam (inject a MockTransport client); else build from settings.
        self._client = http_client or self._http_client()

    # --- capability ---
    @property
    def supported_query_modes(self) -> set[QueryMode]:
        """OpenSearch serves dense, sparse (BM25) and fused hybrid."""
        return {QueryMode.DENSE, QueryMode.SPARSE, QueryMode.HYBRID}

    # --- internals ---
    def _http_client(self) -> httpx.Client:
        """Build an httpx client for OpenSearch admin calls (pipeline setup, mapping, search)."""
        s = self._settings
        auth = (s.opensearch_user, s.opensearch_password) if s.opensearch_user else None
        return httpx.Client(base_url=s.opensearch_url, auth=auth, verify=s.opensearch_verify_certs, timeout=10.0)

    def _vector_store(self, index: str, dim: int) -> OpensearchVectorStore:
        """Create (or attach to) the OpenSearch index and return a hybrid-enabled vector store."""
        s = self._settings
        kwargs = {}
        if s.opensearch_user:
            kwargs["http_auth"] = (s.opensearch_user, s.opensearch_password)
        try:
            client = OllenOpensearchVectorClient(
                s.opensearch_url,
                index,
                dim,
                embedding_field="embedding",
                text_field="content",
                search_pipeline=s.opensearch_hybrid_pipeline,
                verify_certs=s.opensearch_verify_certs,
                **kwargs,
            )
            return OpensearchVectorStore(client)
        except Exception as exc:
            raise VectorStoreError(f"Failed to connect to OpenSearch index '{index}': {exc}") from exc

    def _ensure_pipeline(self) -> None:
        """Idempotently PUT the hybrid score-normalization search pipeline (min_max + weighted mean)."""
        s = self._settings
        body = {
            "description": "ollen-rag hybrid search: BM25 + kNN score normalization",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            # Weight order matches OpenSearch hybrid sub-queries: [sparse/BM25, dense/kNN]
                            "parameters": {"weights": [s.hybrid_sparse_weight, s.hybrid_dense_weight]},
                        },
                    }
                }
            ],
        }
        try:
            response = self._client.put(f"/_search/pipeline/{s.opensearch_hybrid_pipeline}", json=body)
            response.raise_for_status()
            log.info("hybrid pipeline '%s' ensured", s.opensearch_hybrid_pipeline)
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to create hybrid search pipeline: {exc}") from exc

    # --- lifecycle ---
    def warmup(self) -> None:
        """Startup priming: ensure the shared hybrid pipeline exists."""
        self._ensure_pipeline()

    def ensure_ready(self, index: str, dim: int) -> None:
        """Ensure the hybrid pipeline and create/attach the index (client construction creates it)."""
        self._ensure_pipeline()
        self._vector_store(index, dim)

    # --- ingest ---
    def add_nodes(self, index: str, nodes: list[BaseNode]) -> None:
        """Write already-embedded nodes; dim is inferred from the first node's embedding."""
        if not nodes:
            return
        dim = len(nodes[0].get_embedding())
        self._vector_store(index, dim).add(nodes)

    # --- retrieve ---
    def retrieve(self, index, query_str, query_embedding, mode, top_k, raw_filters, filter_condition):
        """Query the index in the given mode; return the raw scored NodeWithScore list.

        The caller supplies the query embedding (resolved from the index's recorded model),
        so this bypasses VectorStoreIndex and queries the store directly.
        """
        store = self._vector_store(index, len(query_embedding))
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_str,
            mode=_MODE_MAP[mode],
            similarity_top_k=top_k,
            filters=_build_metadata_filters(raw_filters, filter_condition),
        )
        try:
            result = store.query(query)
        except Exception as exc:
            raise VectorStoreError(f"Retrieval failed on index '{index}': {exc}") from exc
        sims = result.similarities or [None] * len(result.nodes)
        return [NodeWithScore(node=node, score=score) for node, score in zip(result.nodes, sims)]

    # --- config / meta ---
    def get_index_dim(self, index: str) -> int | None:
        """Return the existing index's vector dimension, or None if the index doesn't exist yet."""
        try:
            response = self._client.get(f"/{index}/_mapping")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            mapping = response.json()
            return mapping[index]["mappings"]["properties"]["embedding"]["dimension"]
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to read mapping for index '{index}': {exc}") from exc

    def set_index_meta(self, index: str, embedding_provider: str, embedding_model: str, chunking: dict) -> None:
        """Record the full build config of an index in the mapping's _meta: embedding provider/model
        at the top level plus the chunking config (strategy + only the relevant knobs). One index = one config."""
        body = {"_meta": {"embedding_provider": embedding_provider, "embedding_model": embedding_model, "chunking": chunking}}
        try:
            response = self._client.put(f"/{index}/_mapping", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to record index metadata for index '{index}': {exc}") from exc

    def get_index_meta(self, index: str) -> dict | None:
        """Return the whole recorded _meta block (embedding + chunking) for this index, or None if the
        index doesn't exist or has no recorded meta (legacy index)."""
        try:
            response = self._client.get(f"/{index}/_mapping")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            mapping = response.json()
            meta = mapping[index]["mappings"].get("_meta")
            return meta or None
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to read mapping for index '{index}': {exc}") from exc

    # --- introspection / admin ---
    def list_indices(self) -> list[dict]:
        """Return app-owned indices with document counts. Starts from _cat/indices, drops OpenSearch
        system indices (names starting with '.'), then keeps only indices carrying our build
        signature in the mapping _meta (embedding_provider). This filters out foreign indices created
        by plugins on the same cluster (e.g. Query Insights' 'top_queries-*'), which otherwise show
        up in the console and 500 when browsed because they lack our document/bucket schema."""
        try:
            response = self._client.get("/_cat/indices", params={"format": "json"})
            response.raise_for_status()
            candidates = [ix for ix in response.json() if not str(ix.get("index", "")).startswith(".")]
            if not candidates:
                return []
            # One bulk _mapping call for all candidates; keep those whose _meta records our build config.
            names = ",".join(str(ix["index"]) for ix in candidates)
            mapping_resp = self._client.get(f"/{names}/_mapping")
            mapping_resp.raise_for_status()
            mappings = mapping_resp.json()
            owned = {
                name for name, m in mappings.items()
                if (m.get("mappings", {}).get("_meta") or {}).get("embedding_provider")
            }
            return [ix for ix in candidates if str(ix.get("index", "")) in owned]
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to list indices: {exc}") from exc

    def get_index_documents(self, index: str, offset: int = 0, limit: int = 20, bucket: str | None = None) -> dict:
        """Paginate raw stored documents (content + metadata) for browsing/inspection, optionally
        scoped to a single bucket.

        Excludes the dense embedding vector and llama-index's internal `_node_content`
        bookkeeping blob so the UI stays readable.
        """
        body_query: dict = {
            "_source": {"excludes": ["embedding", "metadata._node_content"]},
            "sort": [{"metadata.file_name.keyword": {"order": "asc", "unmapped_type": "keyword"}}, "_id"],
        }
        if bucket:
            body_query["query"] = {"term": {"metadata.bucket.keyword": bucket}}
        try:
            response = self._client.post(
                f"/{index}/_search",
                params={"from": offset, "size": limit},
                json=body_query,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to fetch documents for '{index}': {exc}") from exc
        return {
            "total": body["hits"]["total"]["value"],
            "documents": [
                {"id": hit["_id"], "content": hit["_source"].get("content", ""), "metadata": hit["_source"].get("metadata", {})}
                for hit in body["hits"]["hits"]
            ],
        }

    def list_buckets(self, index: str) -> list[str]:
        """Return distinct 'bucket' metadata values present in an index, for preloading UI dropdowns."""
        try:
            response = self._client.post(
                f"/{index}/_search",
                params={"size": 0},
                json={"aggs": {"buckets": {"terms": {"field": "metadata.bucket.keyword", "size": 200}}}},
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to list buckets for '{index}': {exc}") from exc
        return [b["key"] for b in body.get("aggregations", {}).get("buckets", {}).get("buckets", [])]

    def list_bucket_files(self, index: str) -> dict[str, list[str]]:
        """Map each bucket in an index to the distinct document file_names it contains.

        Powers the KB "add to existing" UI so the user sees which docs already live in an
        index+bucket and avoids re-uploading them. Nested terms aggregation: bucket -> file_name.
        """
        try:
            response = self._client.post(
                f"/{index}/_search",
                params={"size": 0},
                json={"aggs": {"buckets": {"terms": {"field": "metadata.bucket.keyword", "size": 200},
                                           "aggs": {"files": {"terms": {"field": "metadata.file_name.keyword", "size": 500}}}}}},
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to list bucket files for '{index}': {exc}") from exc
        out: dict[str, list[str]] = {}
        for b in body.get("aggregations", {}).get("buckets", {}).get("buckets", []):
            out[b["key"]] = [f["key"] for f in b.get("files", {}).get("buckets", [])]
        return out

    def find_duplicate_file(self, index: str, file_hash: str, bucket: str | None) -> str | None:
        """Return the file_name of an already-indexed document with the same content hash, or None.

        Dedup scope is hash AND bucket (bucket separation is a service invariant): the same
        file uploaded into a different bucket is NOT a duplicate. Uploads without a bucket
        dedupe only against other bucketless documents. A missing index counts as no duplicate.
        """
        query: dict = {"bool": {"must": [{"term": {"metadata.file_hash.keyword": file_hash}}]}}
        if bucket is not None:
            query["bool"]["must"].append({"term": {"metadata.bucket.keyword": bucket}})
        else:
            query["bool"]["must_not"] = [{"exists": {"field": "metadata.bucket"}}]
        try:
            response = self._client.post(
                f"/{index}/_search",
                params={"size": 1},
                json={"query": query, "_source": ["metadata.file_name"]},
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            hits = response.json()["hits"]["hits"]
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Duplicate check failed for '{index}': {exc}") from exc
        if not hits:
            return None
        duplicate_of = hits[0]["_source"].get("metadata", {}).get("file_name", "")
        log.debug("dedup hit in %s: %s", index, duplicate_of)
        return duplicate_of

    def delete_index(self, index: str) -> None:
        """Permanently delete an index and all its documents. Irreversible."""
        try:
            response = self._client.delete(f"/{index}")
            if response.status_code not in (200, 404):
                response.raise_for_status()
            log.info("index deleted: %s", index)
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to delete index '{index}': {exc}") from exc

    def delete_bucket(self, index: str, bucket: str) -> int:
        """Delete every document in `index` whose metadata.bucket == `bucket`.
        Returns the number of documents deleted. Idempotent: a missing index or
        bucket deletes nothing and returns 0."""
        try:
            response = self._client.post(
                f"/{index}/_delete_by_query",
                params={"refresh": "true"},
                json={"query": {"term": {"metadata.bucket.keyword": bucket}}},
            )
            if response.status_code == 404:
                return 0
            response.raise_for_status()
            deleted = int(response.json().get("deleted", 0))
            log.info("bucket deleted: %s/%s (%d docs)", index, bucket, deleted)
            return deleted
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"Failed to delete bucket '{bucket}' from '{index}': {exc}") from exc