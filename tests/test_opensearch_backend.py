"""Tests for OpenSearchBackend: admin/meta/dedup via MockTransport, retrieve/add via fake stores."""
import json
import httpx
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQueryResult
from src.exceptions import VectorStoreError
from src.factories.vector_store import QueryMode
from src.providers.vector_stores import opensearch as os_mod
from src.providers.vector_stores.opensearch import OllenOpensearchVectorClient, OpenSearchBackend
from src.settings import Settings

SETTINGS = Settings(_env_file=None)

def _backend(handler) -> OpenSearchBackend:
    """OpenSearchBackend wired to a MockTransport request handler."""
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=SETTINGS.opensearch_url)
    return OpenSearchBackend(SETTINGS, http_client=client)

# --- capability ---

def test_supported_modes_are_all_three():
    backend = _backend(lambda r: httpx.Response(200))
    assert backend.supported_query_modes == {QueryMode.DENSE, QueryMode.SPARSE, QueryMode.HYBRID}

# --- pipeline / lifecycle ---

def test_warmup_puts_pipeline_config():
    captured = {}
    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"acknowledged": True})
    _backend(handler).warmup()
    assert captured["path"] == f"/_search/pipeline/{SETTINGS.opensearch_hybrid_pipeline}"
    assert "normalization-processor" in captured["body"]
    assert "min_max" in captured["body"]

def test_warmup_failure_raises():
    with pytest.raises(VectorStoreError):
        _backend(lambda r: httpx.Response(500, json={"error": "boom"})).warmup()

def test_vector_store_builds_ollen_client(monkeypatch):
    captured = {}
    class _StubClient:
        # Stands in for OllenOpensearchVectorClient to avoid a live cluster
        def __init__(self, endpoint, index, dim, **kwargs):
            captured.update(endpoint=endpoint, index=index, dim=dim, **kwargs)
    class _StubStore:
        def __init__(self, client):
            self.client = client
    monkeypatch.setattr(os_mod, "OllenOpensearchVectorClient", _StubClient)
    monkeypatch.setattr(os_mod, "OpensearchVectorStore", _StubStore)
    store = _backend(lambda r: httpx.Response(200))._vector_store("ollen_rag_sentence", 768)
    assert isinstance(store.client, _StubClient)
    assert captured["index"] == "ollen_rag_sentence"
    assert captured["dim"] == 768
    assert captured["search_pipeline"] == SETTINGS.opensearch_hybrid_pipeline
    assert captured["text_field"] == "content"

# --- ingest ---

def test_add_nodes_writes_to_store(monkeypatch):
    recorded = {}
    class _StubStore:
        def add(self, nodes):
            recorded["nodes"] = nodes
    backend = _backend(lambda r: httpx.Response(200))
    monkeypatch.setattr(backend, "_vector_store", lambda index, dim: _StubStore())
    node = TextNode(text="hi", embedding=[0.1, 0.2, 0.3])
    backend.add_nodes("ollen_rag_sentence", [node])
    assert recorded["nodes"] == [node]

def test_add_nodes_noop_on_empty(monkeypatch):
    backend = _backend(lambda r: httpx.Response(200))
    monkeypatch.setattr(backend, "_vector_store", lambda index, dim: pytest.fail("should not build store"))
    backend.add_nodes("ollen_rag_sentence", [])  # must not raise / build a store

# --- retrieve ---

def test_retrieve_maps_mode_and_wraps_nodes(monkeypatch):
    backend = _backend(lambda r: httpx.Response(200))
    captured = {}
    class FakeStore:
        def query(self, q):
            captured["mode"] = q.mode
            captured["k"] = q.similarity_top_k
            return VectorStoreQueryResult(nodes=[TextNode(text="hit")], similarities=[0.9], ids=["n1"])
    monkeypatch.setattr(backend, "_vector_store", lambda index, dim: FakeStore())
    nodes = backend.retrieve("ollen_rag_sentence", "q", [0.1, 0.2], QueryMode.HYBRID, 5, None, "and")
    assert nodes[0].score == 0.9
    assert captured["mode"] == VectorStoreQueryMode.HYBRID
    assert captured["k"] == 5

def test_retrieve_wraps_failure(monkeypatch):
    backend = _backend(lambda r: httpx.Response(200))
    class BoomStore:
        def query(self, q):
            raise RuntimeError("cluster down")
    monkeypatch.setattr(backend, "_vector_store", lambda index, dim: BoomStore())
    with pytest.raises(VectorStoreError):
        backend.retrieve("ollen_rag_sentence", "q", [0.1], QueryMode.DENSE, 5, None, "and")

# --- meta / dim ---

def test_set_index_meta_puts_mapping():
    captured = {}
    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"acknowledged": True})
    _backend(handler).set_index_meta(
        "ollen_rag_sentence", "fastembed", "BAAI/bge-small-en-v1.5",
        {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
    )
    assert captured["path"] == "/ollen_rag_sentence/_mapping"
    assert "fastembed" in captured["body"]
    assert "_meta" in captured["body"]
    assert "chunk_size" in captured["body"]

def test_get_index_meta_returns_full_block():
    def handler(request):
        return httpx.Response(200, json={"ollen_rag_sentence": {"mappings": {
            "_meta": {"embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-en-v1.5",
                      "chunking": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}},
            "properties": {},
        }}})
    result = _backend(handler).get_index_meta("ollen_rag_sentence")
    assert result["chunking"] == {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}

def test_get_index_meta_missing_meta_returns_none():
    def handler(request):
        return httpx.Response(200, json={"ollen_rag_sentence": {"mappings": {"properties": {}}}})
    assert _backend(handler).get_index_meta("ollen_rag_sentence") is None

def test_get_index_meta_missing_index_returns_none():
    assert _backend(lambda r: httpx.Response(404)).get_index_meta("ollen_rag_sentence") is None

def test_get_index_dim_returns_existing_dim():
    def handler(request):
        assert request.url.path == "/ollen_rag_sentence/_mapping"
        return httpx.Response(200, json={"ollen_rag_sentence": {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 768}}}}})
    assert _backend(handler).get_index_dim("ollen_rag_sentence") == 768

def test_get_index_dim_missing_index_returns_none():
    assert _backend(lambda r: httpx.Response(404)).get_index_dim("ollen_rag_sentence") is None

def test_get_index_dim_error_raises():
    with pytest.raises(VectorStoreError):
        _backend(lambda r: httpx.Response(500, json={"error": "boom"})).get_index_dim("ollen_rag_sentence")

# --- admin ---

def test_list_indices_keeps_owned_hides_system_and_foreign():
    # _cat lists everything; the follow-up _mapping call keeps only indices whose _meta carries our
    # build signature. System (dot) indices are dropped before the mapping call; foreign plugin
    # indices (no embedding_provider in _meta) are dropped after it.
    def handler(request):
        if request.url.path == "/_cat/indices":
            return httpx.Response(200, json=[
                {"index": "sentence", "docs.count": "10"},
                {"index": "custom_name", "docs.count": "3"},
                {"index": "top_queries-2026.07.13-1", "docs.count": "5"},  # foreign plugin index
                {"index": ".opensearch-observability", "docs.count": "1"},  # system → hidden pre-mapping
            ])
        # Bulk mapping lookup for the non-system candidates.
        assert request.url.path == "/sentence,custom_name,top_queries-2026.07.13-1/_mapping"
        return httpx.Response(200, json={
            "sentence": {"mappings": {"_meta": {"embedding_provider": "fastembed"}}},
            "custom_name": {"mappings": {"_meta": {"embedding_provider": "watsonx"}}},
            "top_queries-2026.07.13-1": {"mappings": {}},  # no _meta → foreign, dropped
        })
    listed = [ix["index"] for ix in _backend(handler).list_indices()]
    assert listed == ["sentence", "custom_name"]

def test_get_index_documents_excludes_embedding_and_paginates():
    def handler(request):
        assert request.url.path == "/ollen_rag_sentence/_search"
        assert request.url.params["from"] == "5"
        assert request.url.params["size"] == "2"
        return httpx.Response(200, json={"hits": {"total": {"value": 1},
            "hits": [{"_id": "n1", "_source": {"content": "ciao", "metadata": {"bucket": "soc"}}}]}})
    result = _backend(handler).get_index_documents("ollen_rag_sentence", offset=5, limit=2)
    assert result["total"] == 1
    assert result["documents"][0] == {"id": "n1", "content": "ciao", "metadata": {"bucket": "soc"}}

def test_list_buckets():
    def handler(request):
        return httpx.Response(200, json={"aggregations": {"buckets": {"buckets": [
            {"key": "soc", "doc_count": 5}, {"key": "hr", "doc_count": 2}]}}})
    assert _backend(handler).list_buckets("ollen_rag_sentence") == ["soc", "hr"]

def test_list_bucket_files_maps_bucket_to_filenames():
    def handler(request):
        return httpx.Response(200, json={"aggregations": {"buckets": {"buckets": [
            {"key": "soc", "files": {"buckets": [{"key": "a.pdf"}, {"key": "b.pdf"}]}},
            {"key": "hr", "files": {"buckets": [{"key": "policy.pdf"}]}}]}}})
    assert _backend(handler).list_bucket_files("ollen_rag_sentence") == {"soc": ["a.pdf", "b.pdf"], "hr": ["policy.pdf"]}

def test_delete_index_calls_delete():
    captured = {}
    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"acknowledged": True})
    _backend(handler).delete_index("ollen_rag_sentence")
    assert captured == {"method": "DELETE", "path": "/ollen_rag_sentence"}

def test_delete_index_tolerates_already_missing():
    _backend(lambda r: httpx.Response(404)).delete_index("ollen_rag_sentence")  # must not raise

# --- delete bucket ---

def test_delete_bucket_calls_delete_by_query():
    captured = {}
    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"deleted": 3})
    n = _backend(handler).delete_bucket("ollen_rag_sentence", "soc")
    assert n == 3
    assert captured["method"] == "POST"
    assert captured["path"] == "/ollen_rag_sentence/_delete_by_query"
    assert captured["body"] == {"query": {"term": {"metadata.bucket.keyword": "soc"}}}

def test_delete_bucket_missing_index_returns_zero():
    assert _backend(lambda r: httpx.Response(404)).delete_bucket("ollen_rag_sentence", "soc") == 0

# --- dedup ---

def test_find_duplicate_file_found_same_bucket():
    captured = {}
    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"hits": {"total": {"value": 1},
            "hits": [{"_id": "n1", "_source": {"metadata": {"file_name": "old.pdf"}}}]}})
    result = _backend(handler).find_duplicate_file("ollen_rag_sentence", "abc123", "soc")
    assert result == "old.pdf"
    assert captured["path"] == "/ollen_rag_sentence/_search"
    # Dedup scope is hash AND bucket (bucket separation invariant)
    assert "metadata.file_hash.keyword" in captured["body"] and "abc123" in captured["body"]
    assert "metadata.bucket.keyword" in captured["body"] and "soc" in captured["body"]

def test_find_duplicate_file_no_bucket_scopes_to_bucketless_docs():
    captured = {}
    def handler(request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"hits": {"total": {"value": 0}, "hits": []}})
    result = _backend(handler).find_duplicate_file("ollen_rag_sentence", "abc123", None)
    assert result is None
    # Same hash in some bucket must NOT block a bucketless upload: query excludes bucketed docs
    assert "must_not" in captured["body"] and "metadata.bucket" in captured["body"]

def test_find_duplicate_file_missing_index_returns_none():
    assert _backend(lambda r: httpx.Response(404)).find_duplicate_file("ollen_rag_sentence", "abc123", "soc") is None

def test_find_duplicate_file_error_raises():
    with pytest.raises(VectorStoreError):
        _backend(lambda r: httpx.Response(500, json={"error": "boom"})).find_duplicate_file("ollen_rag_sentence", "abc123", "soc")

# --- lexical (BM25 keyword widening) ---

def _lexical_client():
    # _lexical_search_query needs no connection state; skip the network-touching __init__
    return OllenOpensearchVectorClient.__new__(OllenOpensearchVectorClient)

def test_lexical_query_widens_to_keywords_field():
    query = _lexical_client()._lexical_search_query("content", "codici colore", 7)
    must = query["query"]["bool"]["must"]
    assert must == {"multi_match": {"query": "codici colore", "fields": ["content", "metadata.keywords^2"]}}
    assert query["size"] == 7
    assert "filter" not in query["query"]["bool"]

def test_lexical_query_keeps_metadata_filters():
    from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
    filters = MetadataFilters(filters=[MetadataFilter(key="bucket", value="soc")])
    query = _lexical_client()._lexical_search_query("content", "q", 5, filters=filters)
    # bucket separation invariant: filters must survive the multi_match rewrite
    assert query["query"]["bool"]["filter"], "metadata filters were dropped"
    assert "soc" in str(query["query"]["bool"]["filter"])

def test_lexical_query_respects_excluded_source_fields():
    query = _lexical_client()._lexical_search_query("content", "q", 5, excluded_source_fields=["embedding"])
    assert query["_source"] == {"exclude": ["embedding"]}

def test_ollen_client_overrides_lexical_leg():
    from llama_index.vector_stores.opensearch import OpensearchVectorClient
    # The subclass must actually override the lexical leg the hybrid query delegates to
    assert OllenOpensearchVectorClient._lexical_search_query is not OpensearchVectorClient._lexical_search_query