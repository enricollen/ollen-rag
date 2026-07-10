"""API tests with the RAG core mocked out; exercises routing, schemas, error mapping, jobs."""
import io
import pytest
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode
from src.api import routes
from src.exceptions import ParsingError

@pytest.fixture
def client():
    # Import inside fixture so create_app picks up test env from conftest.
    # Use TestClient as a context manager so the combined app/MCP lifespan
    # actually runs startup (needed to init the MCP session manager for /mcp).
    from app import create_app
    with TestClient(create_app()) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_strategies(client):
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200
    assert "semantic" in response.json()["strategies"]

def test_ingest_creates_job_and_completes(client, monkeypatch):
    # Background task runs synchronously under TestClient, so the job finishes immediately.
    # Patch ingest_document in the ingestion module: routes delegates to
    # ingestion.run_ingestion_job, which resolves the callable there at run time.
    from src.rag import ingestion
    monkeypatch.setattr(
        ingestion, "ingest_document",
        lambda path, strategy=None, index_name=None, extra_metadata=None, file_name=None, enrich_keywords=None,
               embedding_provider=None, embedding_model=None, chunk_params=None, progress_cb=None: {
            "index": "ollen_rag_sentence", "strategy": "sentence", "num_documents": 1, "num_nodes": 3,
        },
    )
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")},
        data={"strategy": "sentence", "metadata": '{"team": "soc"}'},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get(f"/api/v1/ingest/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["result"]["num_nodes"] == 3

def test_ingest_forwards_embedding_fields(client, monkeypatch):
    from src.rag import ingestion
    captured = {}
    def fake_ingest_document(path, strategy=None, index_name=None, extra_metadata=None, file_name=None,
                              enrich_keywords=None, embedding_provider=None, embedding_model=None, chunk_params=None, progress_cb=None):
        captured["embedding_provider"] = embedding_provider
        captured["embedding_model"] = embedding_model
        return {"index": "ollen_rag_sentence", "strategy": "sentence", "num_documents": 1, "num_nodes": 3}
    monkeypatch.setattr(ingestion, "ingest_document", fake_ingest_document)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")},
        data={"strategy": "sentence", "embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-en-v1.5"},
    )
    assert response.status_code == 202
    assert captured["embedding_provider"] == "fastembed"
    assert captured["embedding_model"] == "BAAI/bge-small-en-v1.5"

def test_ingest_unknown_embedding_provider_fails_job(client):
    # ingest_document itself validates (via ValueError); route only forwards the fields,
    # so the failure surfaces as a failed job, not an HTTP-level error.
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.txt", io.BytesIO(b"hello world"), "text/plain")},
        data={"strategy": "sentence", "embedding_provider": "banana"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get(f"/api/v1/ingest/{job_id}")
    assert status.json()["status"] == "failed"
    assert "banana" in status.json()["detail"]

def test_ingest_invalid_metadata_json(client):
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b"x"), "application/pdf")},
        data={"metadata": "not-json"},
    )
    assert response.status_code == 400

def test_ingest_job_not_found(client):
    assert client.get("/api/v1/ingest/nope").status_code == 404

def test_retrieve_serializes_numpy_score(client, monkeypatch):
    # SentenceTransformerRerank assigns numpy.float32 scores; JSON encoding
    # must not choke on them (regression: caught only against a real reranker).
    import numpy
    node = NodeWithScore(node=TextNode(text="ciao", metadata={}), score=numpy.float32(0.87))
    monkeypatch.setattr(routes, "retrieve_debug", lambda query, **kwargs: {"bm25": [], "dense": [], "hybrid": [], "reranked": [node]})
    response = client.post("/api/v1/retrieve", json={"query": "test?"})
    assert response.status_code == 200
    assert response.json()["nodes"][0]["score"] == pytest.approx(0.87, abs=1e-6)

def test_retrieve(client, monkeypatch):
    bm25_node = NodeWithScore(node=TextNode(text="bm25-hit", metadata={}), score=3.1)
    dense_node = NodeWithScore(node=TextNode(text="dense-hit", metadata={}), score=0.5)
    reranked_node = NodeWithScore(node=TextNode(text="ciao", metadata={"file_name": "a.pdf"}), score=0.9)
    captured = {}
    def fake_retrieve_debug(query, **kwargs):
        captured.update(query=query, **kwargs)
        return {"bm25": [bm25_node], "dense": [dense_node], "hybrid": [], "reranked": [reranked_node]}
    monkeypatch.setattr(routes, "retrieve_debug", fake_retrieve_debug)
    response = client.post("/api/v1/retrieve", json={
        "query": "test?", "strategy": "semantic",
        "filters": [{"key": "team", "value": "soc"}], "filter_condition": "and",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["text"] == "ciao"
    assert body["nodes"][0]["score"] == 0.9
    assert body["bm25_nodes"][0]["text"] == "bm25-hit"
    assert body["dense_nodes"][0]["text"] == "dense-hit"
    assert captured["strategy"] == "semantic"
    assert captured["raw_filters"] == [{"key": "team", "value": "soc", "operator": "=="}]

def test_query(client, monkeypatch):
    monkeypatch.setattr(
        routes, "generate",
        lambda query, **kwargs: {"answer": "Risposta [1]", "sources": [{"id": 1, "text": "t", "score": 0.9, "metadata": {}}]},
    )
    response = client.post("/api/v1/query", json={"query": "test?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Risposta [1]"
    assert response.json()["sources"][0]["id"] == 1

def test_retrieve_forwards_reranker_model(client, monkeypatch):
    captured = {}
    def fake_retrieve_debug(query, **kwargs):
        captured["reranker_model"] = kwargs.get("reranker_model")
        return {"bm25": [], "dense": [], "hybrid": [], "reranked": []}
    monkeypatch.setattr(routes, "retrieve_debug", fake_retrieve_debug)
    response = client.post("/api/v1/retrieve", json={"query": "x", "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    assert response.status_code == 200
    assert captured["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

def test_query_forwards_reranker_model(client, monkeypatch):
    captured = {}
    def fake_generate(query, **kwargs):
        captured["reranker_model"] = kwargs.get("reranker_model")
        return {"answer": "ok", "sources": []}
    monkeypatch.setattr(routes, "generate", fake_generate)
    response = client.post("/api/v1/query", json={"query": "x", "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    assert response.status_code == 200
    assert captured["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

def test_config_exposes_reranker_model_choices(client):
    body = client.get("/api/v1/config").json()
    assert "default" in body["reranker_model_choices"]
    assert body["reranker_model_choices"]["default"] == "models/reranker"

def test_domain_error_maps_to_http(client, monkeypatch):
    def boom(query, **kwargs):
        raise ParsingError("bad file")
    monkeypatch.setattr(routes, "retrieve_debug", boom)
    response = client.post("/api/v1/retrieve", json={"query": "x"})
    assert response.status_code == 422
    assert response.json()["error_code"] == "PARSING_ERROR"

class _AdminBackend:
    """Configurable fake VectorStoreBackend for admin-endpoint tests."""
    def __init__(self):
        self.indices = []
        self.buckets = []
        self.bucket_files = {}
        self.meta = None
        self.dim = None
        self.documents = {"total": 0, "documents": []}
        self.delete_error = None
        self.deleted = None
    def list_indices(self):
        return self.indices
    def list_buckets(self, index):
        return self.buckets
    def list_bucket_files(self, index):
        return self.bucket_files
    def get_index_meta(self, index):
        return self.meta
    def get_index_dim(self, index):
        return self.dim
    def get_index_documents(self, index, offset=0, limit=20, bucket=None):
        return self.documents
    def delete_index(self, index):
        if self.delete_error:
            raise self.delete_error
        self.deleted = index

def _use_backend(monkeypatch, backend):
    """Patch the routes' backend accessor to return *backend*."""
    monkeypatch.setattr(routes, "create_backend", lambda settings=None: backend)
    return backend

def test_indices(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.indices = [{"index": "ollen_rag_sentence", "docs.count": "5"}]
    response = client.get("/api/v1/indices")
    assert response.status_code == 200
    assert response.json()["indices"][0]["index"] == "ollen_rag_sentence"

def test_index_buckets(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.buckets = ["soc", "hr"]
    response = client.get("/api/v1/indices/ollen_rag_sentence/buckets")
    assert response.status_code == 200
    assert response.json()["buckets"] == ["soc", "hr"]

def test_index_embedding_meta_present(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.meta = {"embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-en-v1.5"}
    response = client.get("/api/v1/indices/ollen_rag_sentence/embedding")
    assert response.status_code == 200
    assert response.json() == {"embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-en-v1.5"}

def test_index_embedding_meta_absent(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.meta = None
    response = client.get("/api/v1/indices/ollen_rag_sentence/embedding")
    assert response.status_code == 200
    assert response.json() == {"embedding_provider": None, "embedding_model": None}

def test_index_info_returns_full_config(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.meta = {
        "embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunking": {"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 16},
    }
    backend.dim = 384
    backend.indices = [{"index": "ollen_rag_sentence", "docs.count": "7"}]
    backend.buckets = ["soc"]
    backend.bucket_files = {"soc": ["a.pdf", "b.pdf"]}
    response = client.get("/api/v1/indices/ollen_rag_sentence/info")
    assert response.status_code == 200
    body = response.json()
    assert body["embedding_provider"] == "fastembed"
    assert body["chunking"] == {"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 16}
    assert body["dim"] == 384
    assert body["docs_count"] == 7
    assert body["buckets"] == ["soc"]
    assert body["bucket_files"] == {"soc": ["a.pdf", "b.pdf"]}

def test_index_info_legacy_index_nulls(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())  # all defaults: no meta, no dim, empty lists
    body = client.get("/api/v1/indices/ollen_rag_sentence/info").json()
    assert body["embedding_provider"] is None
    assert body["chunking"] is None
    assert body["docs_count"] is None

def test_config_excludes_secrets(client):
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["embedding_provider"]
    assert "watsonx_apikey" not in body
    assert "opensearch_password" not in body

def test_index_documents(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.documents = {"total": 1, "documents": [{"id": "x1", "content": "ciao", "metadata": {"bucket": "soc"}}]}
    response = client.get("/api/v1/indices/ollen_rag_sentence/documents?offset=0&limit=20")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["content"] == "ciao"

def test_index_delete(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    response = client.delete("/api/v1/indices/ollen_rag_sentence")
    assert response.status_code == 200
    assert response.json() == {"deleted": "ollen_rag_sentence"}
    assert backend.deleted == "ollen_rag_sentence"

def test_index_delete_rejects_unowned_index(client, monkeypatch):
    backend = _use_backend(monkeypatch, _AdminBackend())
    backend.delete_error = ValueError("'some-other-service-index' is not a ollen_rag* index")
    response = client.delete("/api/v1/indices/some-other-service-index")
    assert response.status_code == 400

def test_mcp_mounted(client):
    # The MCP endpoint must exist (405/406/400 acceptable for a plain GET without MCP handshake)
    response = client.get("/mcp/")
    assert response.status_code != 404

def test_retrieve_passes_similarity_threshold(client, monkeypatch):
    """Request-level threshold must reach the rag layer untouched."""
    captured = {}
    def _fake_debug(query, **kwargs):
        captured.update(kwargs)
        return {"bm25": [], "dense": [], "hybrid": [], "reranked": []}
    monkeypatch.setattr(routes, "retrieve_debug", _fake_debug)
    resp = client.post("/api/v1/retrieve", json={"query": "q", "similarity_threshold": 0.3})
    assert resp.status_code == 200
    assert captured["similarity_threshold"] == 0.3

def test_retrieve_rejects_out_of_range_threshold(client):
    resp = client.post("/api/v1/retrieve", json={"query": "q", "similarity_threshold": 1.5})
    assert resp.status_code == 422

def test_config_exposes_similarity_threshold(client):
    assert client.get("/api/v1/config").json()["similarity_threshold"] == 0.0

class _FakeReport:
    """Stands in for EvalReport; the endpoint only calls to_dict()."""
    def to_dict(self):
        return {"params": {}, "overall": {"cases": 1, "hit_rate": 1.0, "recall": 1.0, "mrr": 1.0}, "per_bucket": {}, "cases": []}

def test_eval_endpoint_with_inline_cases(client, monkeypatch):
    monkeypatch.setattr(routes, "evaluate", lambda ds, **kw: _FakeReport())
    resp = client.post("/api/v1/eval/retrieval", json={
        "cases": [{"query": "q", "bucket": "b", "expected": [{"file_name": "a.pdf"}]}],
    })
    assert resp.status_code == 200
    assert resp.json()["overall"]["cases"] == 1

def test_eval_endpoint_unknown_dataset_404(client):
    resp = client.post("/api/v1/eval/retrieval", json={"dataset": "does-not-exist"})
    assert resp.status_code == 404

def test_eval_endpoint_rejects_bad_dataset_name(client):
    resp = client.post("/api/v1/eval/retrieval", json={"dataset": "../../etc/passwd"})
    assert resp.status_code == 422

def test_eval_endpoint_requires_exactly_one_source(client):
    assert client.post("/api/v1/eval/retrieval", json={}).status_code == 422
    assert client.post("/api/v1/eval/retrieval", json={
        "dataset": "x", "cases": [{"query": "q", "bucket": "b", "expected": [{"file_name": "a"}]}],
    }).status_code == 422

def test_eval_endpoint_invalid_case_schema_422(client):
    # missing bucket -> parse_dataset ValueError -> 422
    resp = client.post("/api/v1/eval/retrieval", json={"cases": [{"query": "q", "expected": [{"file_name": "a"}]}]})
    assert resp.status_code == 422

def test_eval_endpoint_expected_missing_file_name_422(client):
    # expected entry without file_name -> parse_dataset ValueError -> 422, not a 500 KeyError
    resp = client.post("/api/v1/eval/retrieval", json={
        "cases": [{"query": "q", "bucket": "b", "expected": [{"contains": "x"}]}],
    })
    assert resp.status_code == 422

def test_retrieve_returns_hybrid_leg_and_retrieval_scores(client, monkeypatch):
    """Reranked nodes carry retrieval_score joined from the hybrid leg; hybrid_nodes exposed."""
    fused = NodeWithScore(node=TextNode(id_="n1", text="t"), score=0.8)
    reranked = NodeWithScore(node=TextNode(id_="n1", text="t"), score=4.2)  # cross-encoder scale
    monkeypatch.setattr(
        routes, "retrieve_debug",
        lambda query, **kwargs: {"bm25": [], "dense": [], "hybrid": [fused], "reranked": [reranked]},
    )
    body = client.post("/api/v1/retrieve", json={"query": "q"}).json()
    assert body["nodes"][0]["score"] == 4.2
    assert body["nodes"][0]["retrieval_score"] == 0.8
    assert body["hybrid_nodes"][0]["score"] == 0.8
    assert "retrieval_score" not in body["hybrid_nodes"][0]

def test_ingest_forwards_enrich_keywords_flag(client, monkeypatch):
    # Form field "enrich_keywords" must reach ingest_document as a real bool
    from src.rag import ingestion
    captured = {}
    def fake_ingest(path, strategy=None, index_name=None, extra_metadata=None, file_name=None, enrich_keywords=None, embedding_provider=None, embedding_model=None, chunk_params=None, progress_cb=None):
        captured["enrich_keywords"] = enrich_keywords
        return {"index": "ollen_rag_sentence", "strategy": "sentence", "num_documents": 1, "num_nodes": 1, "enriched": bool(enrich_keywords)}
    monkeypatch.setattr(ingestion, "ingest_document", fake_ingest)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")},
        data={"strategy": "sentence", "enrich_keywords": "true"},
    )
    assert response.status_code == 202
    assert captured["enrich_keywords"] is True

def test_ingest_enrich_keywords_absent_means_none(client, monkeypatch):
    # No form field -> None -> ingest_document applies the settings default
    from src.rag import ingestion
    captured = {}
    def fake_ingest(path, strategy=None, index_name=None, extra_metadata=None, file_name=None, enrich_keywords=None, embedding_provider=None, embedding_model=None, chunk_params=None, progress_cb=None):
        captured["enrich_keywords"] = enrich_keywords
        return {"index": "ollen_rag_sentence", "strategy": "sentence", "num_documents": 1, "num_nodes": 1, "enriched": False}
    monkeypatch.setattr(ingestion, "ingest_document", fake_ingest)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")},
    )
    assert response.status_code == 202
    assert captured["enrich_keywords"] is None

def test_config_exposes_enrich_keywords(client):
    assert client.get("/api/v1/config").json()["enrich_keywords"] is False

def test_config_exposes_log_level(client):
    assert client.get("/api/v1/config").json()["log_level"] == "INFO"

def test_config_exposes_embedding_model_choices(client):
    body = client.get("/api/v1/config").json()
    assert "fastembed" in body["embedding_model_choices"]
    assert "BAAI/bge-small-en-v1.5" in body["embedding_model_choices"]["fastembed"]

def test_ingest_status_exposes_progress_and_stage(client, monkeypatch):
    from src.rag import ingestion
    monkeypatch.setattr(
        ingestion, "ingest_document",
        lambda path, strategy=None, index_name=None, extra_metadata=None, file_name=None, enrich_keywords=None, embedding_provider=None, embedding_model=None, chunk_params=None, progress_cb=None: {
            "index": "ollen_rag_sentence", "strategy": "sentence", "num_documents": 1, "num_nodes": 1, "enriched": False,
        },
    )
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")},
    )
    status = client.get(f"/api/v1/ingest/{response.json()['job_id']}").json()
    assert status["progress"] == 100
    assert status["stage"] == "done"
