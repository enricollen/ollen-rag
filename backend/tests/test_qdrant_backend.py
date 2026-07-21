"""Tests for QdrantBackend: in-memory QdrantClient (no server), covering capability, node
round-trip, meta reconstruction, admin/browse, bucket aggregation and dedup."""
import pytest
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from src.exceptions import VectorStoreError
from src.factories.vector_store import QueryMode, VectorStoreFactory
from src.providers.vector_stores.qdrant import QdrantBackend
from src.settings import Settings

SETTINGS = Settings(_env_file=None)
IDX = "ollen_rag_qdrant"

def _backend() -> QdrantBackend:
    """QdrantBackend wired to a throwaway in-memory client."""
    return QdrantBackend(SETTINGS, client=QdrantClient(":memory:"))

def _node(text, file_name, bucket=None, file_hash="h", embedding=(0.1, 0.2, 0.3)) -> TextNode:
    """build a TextNode carrying the metadata the backend groups/deduplicates on."""
    meta = {"file_name": file_name, "file_hash": file_hash}
    if bucket is not None:
        meta["bucket"] = bucket
    return TextNode(text=text, metadata=meta, embedding=list(embedding))

# --- registration / capability ---

def test_registered_under_qdrant():
    assert VectorStoreFactory._registry.get("qdrant") is QdrantBackend

def test_supported_modes_dense_only():
    assert _backend().supported_query_modes == {QueryMode.DENSE}

# --- lifecycle / dim ---

def test_ensure_ready_stamps_dim():
    b = _backend()
    b.ensure_ready(IDX, 768)
    assert b.get_index_dim(IDX) == 768

def test_get_index_dim_missing_is_none():
    assert _backend().get_index_dim("ollen_rag_nope") is None

# --- ingest / retrieve ---

def test_add_and_retrieve_dense():
    b = _backend()
    b.add_nodes(IDX, [
        _node("triage red code", "a.pdf", "conosci", embedding=(0.1, 0.2, 0.3)),
        _node("green code info", "b.pdf", "conosci", embedding=(0.9, 0.8, 0.7)),
    ])
    nodes = b.retrieve(IDX, "triage", [0.1, 0.2, 0.3], QueryMode.DENSE, 2, None, "and")
    assert nodes[0].node.metadata["file_name"] == "a.pdf"
    assert nodes[0].score is not None

def test_retrieve_bucket_filter():
    b = _backend()
    b.add_nodes(IDX, [
        _node("x", "a.pdf", "b1", file_hash="h1"),
        _node("y", "b.pdf", "b2", file_hash="h2"),
    ])
    nodes = b.retrieve(IDX, "x", [0.1, 0.2, 0.3], QueryMode.DENSE, 5,
                       [{"key": "bucket", "value": "b1", "operator": "=="}], "and")
    assert {n.node.metadata["file_name"] for n in nodes} == {"a.pdf"}

def test_add_empty_is_noop():
    _backend().add_nodes(IDX, [])  # must not raise

def test_retrieve_missing_index_raises():
    with pytest.raises(VectorStoreError):
        _backend().retrieve("ollen_rag_nope", "q", [0.1, 0.2, 0.3], QueryMode.DENSE, 1, None, "and")

# --- meta ---

def test_meta_roundtrip_reconstructs_nested_shape():
    b = _backend()
    b.ensure_ready(IDX, 3)
    chunking = {"strategy": "sentence", "chunk_size": 512}
    b.set_index_meta(IDX, "watsonx", "ibm/granite-embedding-278m-multilingual", chunking)
    meta = b.get_index_meta(IDX)
    assert meta == {
        "embedding_provider": "watsonx",
        "embedding_model": "ibm/granite-embedding-278m-multilingual",
        "chunking": chunking,
    }

def test_meta_missing_is_none():
    b = _backend()
    b.ensure_ready(IDX, 3)
    assert b.get_index_meta(IDX) is None

def test_set_meta_preserves_dim():
    b = _backend()
    b.ensure_ready(IDX, 3)
    b.set_index_meta(IDX, "watsonx", "m", {"strategy": "token"})
    assert b.get_index_dim(IDX) == 3

def test_meta_point_excluded_from_doc_count():
    b = _backend()
    b.ensure_ready(IDX, 3)
    b.set_index_meta(IDX, "fastembed", "BAAI/bge-small-en-v1.5", {"strategy": "sentence"})
    listed = {d["index"]: d["docs.count"] for d in b.list_indices()}
    assert listed[IDX] == "0"

# --- admin / browse ---

def test_list_indices_includes_all_names():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", "b1")])
    b.ensure_ready("other_index", 3)
    listed = {d["index"]: d["docs.count"] for d in b.list_indices()}
    assert listed == {IDX: "1", "other_index": "0"}

def test_get_documents_hides_internal_keys():
    b = _backend()
    b.add_nodes(IDX, [_node("hello world", "a.pdf", "b1")])
    page = b.get_index_documents(IDX, 0, 20)
    assert page["total"] == 1
    doc = page["documents"][0]
    assert doc["content"] == "hello world"
    assert doc["metadata"]["file_name"] == "a.pdf"
    assert "_node_content" not in doc["metadata"] and "_node_type" not in doc["metadata"]

def test_get_documents_scoped_to_bucket():
    b = _backend()
    b.add_nodes(IDX, [
        _node("one", "a.pdf", "b1", file_hash="h1"),
        _node("two", "b.pdf", "b2", file_hash="h2"),
        _node("three", "c.pdf", "b2", file_hash="h3"),
    ])
    page = b.get_index_documents(IDX, 0, 20, bucket="b2")
    assert page["total"] == 2
    assert {d["metadata"]["file_name"] for d in page["documents"]} == {"b.pdf", "c.pdf"}

# --- bucket aggregation ---

def test_buckets_and_bucket_files():
    b = _backend()
    b.add_nodes(IDX, [
        _node("1", "a.pdf", "b1", file_hash="h1"),
        _node("2", "a.pdf", "b1", file_hash="h1"),
        _node("3", "c.pdf", "b2", file_hash="h3"),
    ])
    assert b.list_buckets(IDX) == ["b1", "b2"]
    assert b.list_bucket_files(IDX) == {"b1": ["a.pdf"], "b2": ["c.pdf"]}

def test_unbucketed_files_and_documents():
    b = _backend()
    b.add_nodes(IDX, [
        _node("1", "a.pdf", bucket=None, file_hash="h1"),
        _node("2", "a.pdf", bucket=None, file_hash="h1"),
        _node("3", "b.pdf", bucket=None, file_hash="h2"),
    ])
    assert b.list_buckets(IDX) == []
    assert b.list_bucket_files(IDX) == {}
    assert b.list_unbucketed_files(IDX) == ["a.pdf", "b.pdf"]
    page = b.get_index_documents(IDX, 0, 20, unbucketed=True)
    assert page["total"] == 3
    assert {d["metadata"]["file_name"] for d in page["documents"]} == {"a.pdf", "b.pdf"}

def test_unbucketed_files_excludes_bucketed_docs():
    b = _backend()
    b.add_nodes(IDX, [
        _node("1", "a.pdf", "b1", file_hash="h1"),
        _node("2", "loose.pdf", bucket=None, file_hash="h2"),
    ])
    assert b.list_unbucketed_files(IDX) == ["loose.pdf"]
    page = b.get_index_documents(IDX, 0, 20, unbucketed=True)
    assert page["total"] == 1
    assert page["documents"][0]["metadata"]["file_name"] == "loose.pdf"

# --- dedup ---

def test_find_duplicate_same_hash_and_bucket():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", "b1", file_hash="hash1")])
    assert b.find_duplicate_file(IDX, "hash1", "b1") == "a.pdf"

def test_find_duplicate_different_bucket_is_none():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", "b1", file_hash="hash1")])
    assert b.find_duplicate_file(IDX, "hash1", "b2") is None

def test_find_duplicate_bucketless_scope():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", bucket=None, file_hash="hash1")])
    assert b.find_duplicate_file(IDX, "hash1", None) == "a.pdf"
    assert b.find_duplicate_file(IDX, "hash1", "b1") is None

def test_find_duplicate_missing_index_is_none():
    assert _backend().find_duplicate_file("ollen_rag_nope", "h", "b1") is None

# --- delete ---

def test_delete_index_removes_collection():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", "b1")])
    b.delete_index(IDX)
    assert b.list_indices() == []

def test_delete_missing_is_idempotent():
    _backend().delete_index("ollen_rag_nope")

def test_delete_bucket_removes_only_that_bucket():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", "b1"), _node("y", "b.pdf", "b2")])
    n = b.delete_bucket(IDX, "b1")
    assert n == 1
    assert b.list_buckets(IDX) == ["b2"]

def test_delete_bucket_missing_bucket_returns_zero():
    b = _backend()
    b.add_nodes(IDX, [_node("x", "a.pdf", "b1")])
    assert b.delete_bucket(IDX, "nope") == 0
    assert b.list_buckets(IDX) == ["b1"]

def test_delete_bucket_missing_index_returns_zero():
    assert _backend().delete_bucket("ollen_rag_nope", "b1") == 0

# --- vectors (visualizer) ---

def test_get_index_vectors_returns_embeddings_text_metadata():
    b = _backend()
    b.add_nodes(IDX, [
        _node("triage red code", "a.pdf", "b1", embedding=(0.1, 0.2, 0.3)),
        _node("green code info", "b.pdf", "b2", embedding=(0.9, 0.8, 0.7)),
    ])
    vecs = b.get_index_vectors(IDX)
    assert len(vecs) == 2
    by_text = {v["text"]: v for v in vecs}
    # qdrant cosine collections store L2-normalized vectors.
    import math
    raw = [0.1, 0.2, 0.3]
    n = math.sqrt(sum(x * x for x in raw))
    assert by_text["triage red code"]["embedding"] == pytest.approx([x / n for x in raw], abs=1e-5)
    assert by_text["triage red code"]["metadata"]["bucket"] == "b1"
    assert "_node_content" not in by_text["triage red code"]["metadata"]

def test_get_index_vectors_respects_limit():
    b = _backend()
    b.add_nodes(IDX, [_node(f"t{i}", f"{i}.pdf", "b1", file_hash=f"h{i}") for i in range(5)])
    vecs = b.get_index_vectors(IDX, limit=2)
    assert len(vecs) == 2

def test_get_index_vectors_missing_index_returns_empty():
    assert _backend().get_index_vectors("ollen_rag_nope") == []
