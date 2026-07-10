"""Tests for parsing + ingestion pipeline with all external services mocked."""
import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from src.exceptions import ParsingError
from src.rag import ingestion


class _FakeParseResult:
    """Mimics liteparse ParseResult (text + pages)."""
    def __init__(self, text, num_pages=2):
        self.text = text
        self.pages = [object()] * num_pages


class _FakeLiteParse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def parse(self, path):
        return _FakeParseResult("# Title\n\nSome markdown content about pandas.")


class _FakeLiteParseEmpty(_FakeLiteParse):
    def parse(self, path):
        return _FakeParseResult("   ")


class _FakeBackend:
    """In-memory VectorStoreBackend stand-in: configurable dedup/meta/dim, records writes."""
    def __init__(self):
        self.duplicate = None          # find_duplicate_file return value
        self.meta = None               # get_index_meta return value
        self.dim = None                # get_index_dim return value
        self.added = []                # nodes handed to add_nodes
        self.set_meta_calls = []       # recorded set_index_meta calls
        self.dup_args = None           # last find_duplicate_file args

    def ensure_ready(self, index, dim):
        pass

    def find_duplicate_file(self, index, file_hash, bucket):
        self.dup_args = {"index_name": index, "file_hash": file_hash, "bucket": bucket}
        return self.duplicate

    def get_index_meta(self, index):
        return self.meta

    def get_index_dim(self, index):
        return self.dim

    def add_nodes(self, index, nodes):
        self.added.extend(nodes)

    def set_index_meta(self, index, embedding_provider, embedding_model, chunking):
        self.set_meta_calls.append({
            "index_name": index, "embedding_provider": embedding_provider,
            "embedding_model": embedding_model, "chunking": chunking,
        })


@pytest.fixture
def mocked_stack(monkeypatch):
    """Replace every external dependency of ingest_document with in-memory fakes.

    Returns the fake backend so tests can configure dedup/meta/dim and inspect writes.
    """
    backend = _FakeBackend()
    monkeypatch.setattr(ingestion, "LiteParse", _FakeLiteParse)
    monkeypatch.setattr(ingestion, "create_embedding_model", lambda settings=None: MockEmbedding(embed_dim=8))
    monkeypatch.setattr(ingestion, "get_embedding_dim", lambda m: 8)
    monkeypatch.setattr(ingestion, "create_backend", lambda settings=None: backend)
    return backend


@pytest.fixture
def sample_file(tmp_path):
    # ingest_document hashes real file bytes, so tests need an actual file on disk
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"fake pdf bytes")
    return str(f)


def test_parse_file_returns_document(monkeypatch):
    monkeypatch.setattr(ingestion, "LiteParse", _FakeLiteParse)
    docs = ingestion.parse_file("/tmp/fake.pdf")
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["file_name"] == "fake.pdf"
    assert docs[0].metadata["num_pages"] == 2


def test_parse_file_empty_raises(monkeypatch):
    monkeypatch.setattr(ingestion, "LiteParse", _FakeLiteParseEmpty)
    with pytest.raises(ParsingError):
        ingestion.parse_file("/tmp/fake.pdf")


def test_ingest_document(mocked_stack, sample_file):
    result = ingestion.ingest_document(sample_file, strategy="sentence", extra_metadata={"team": "soc"})
    assert result["index"] == "ollen_rag_sentence"
    assert result["strategy"] == "sentence"
    assert result["num_documents"] == 1
    assert result["num_nodes"] >= 1
    assert result["file_hash"] == ingestion.compute_file_hash(sample_file)
    assert mocked_stack.added, "pipeline stored no nodes"


def test_ingest_document_invalid_strategy(mocked_stack):
    with pytest.raises(ValueError):
        ingestion.ingest_document("/tmp/fake.pdf", strategy="banana")


def test_ingest_document_embedding_override_passed_to_settings(mocked_stack, sample_file, monkeypatch):
    captured = {}
    def fake_create_embedding_model(settings=None):
        captured["provider"] = settings.embedding_provider
        captured["fastembed_model_name"] = settings.fastembed_model_name
        return MockEmbedding(embed_dim=8)
    monkeypatch.setattr(ingestion, "create_embedding_model", fake_create_embedding_model)
    ingestion.ingest_document(
        sample_file, strategy="sentence",
        embedding_provider="fastembed", embedding_model="BAAI/bge-large-en-v1.5",
    )
    assert captured["provider"] == "fastembed"
    assert captured["fastembed_model_name"] == "BAAI/bge-large-en-v1.5"


def test_ingest_document_no_override_keeps_default_settings(mocked_stack, sample_file, monkeypatch):
    captured = {}
    def fake_create_embedding_model(settings=None):
        captured["provider"] = settings.embedding_provider
        return MockEmbedding(embed_dim=8)
    monkeypatch.setattr(ingestion, "create_embedding_model", fake_create_embedding_model)
    ingestion.ingest_document(sample_file, strategy="sentence")
    assert captured["provider"] == "watsonx"  # Settings() default, unchanged behavior


def test_ingest_document_rejects_unknown_provider(mocked_stack, sample_file):
    with pytest.raises(ValueError):
        ingestion.ingest_document(sample_file, strategy="sentence", embedding_provider="banana")


def test_ingest_document_rejects_dimension_mismatch(mocked_stack, sample_file, monkeypatch):
    mocked_stack.dim = 768  # existing index recorded 768-dim vectors
    monkeypatch.setattr(ingestion, "get_embedding_dim", lambda m: 8)
    with pytest.raises(ValueError, match="768"):
        ingestion.ingest_document(sample_file, strategy="sentence")


def test_ingest_document_no_existing_index_skips_dim_check(mocked_stack, sample_file):
    mocked_stack.dim = None  # fresh index
    result = ingestion.ingest_document(sample_file, strategy="sentence")
    assert result["num_nodes"] >= 1


def test_compute_file_hash_is_sha256_of_bytes(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    # sha256("hello")
    assert ingestion.compute_file_hash(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_ingest_document_stamps_file_hash_metadata(mocked_stack, sample_file, monkeypatch):
    # file_hash must land in Document metadata so it propagates to every chunk in the store
    captured = {}
    real_parse = ingestion.parse_file
    def spy_parse(path, file_name=None):
        docs = real_parse(path, file_name)
        captured["docs"] = docs
        return docs
    monkeypatch.setattr(ingestion, "parse_file", spy_parse)
    result = ingestion.ingest_document(sample_file, strategy="sentence", extra_metadata={"bucket": "soc"})
    assert captured["docs"][0].metadata["file_hash"] == result["file_hash"]
    assert captured["docs"][0].metadata["bucket"] == "soc"


def test_ingest_document_skips_duplicate(mocked_stack, sample_file):
    mocked_stack.duplicate = "existing.pdf"
    result = ingestion.ingest_document(sample_file, strategy="sentence", extra_metadata={"bucket": "soc"})
    assert result["skipped_duplicate"] is True
    assert result["duplicate_of"] == "existing.pdf"
    assert result["num_nodes"] == 0
    assert result["num_documents"] == 0
    # Dedup check must be bucket-scoped against the target index
    assert mocked_stack.dup_args["index_name"] == "ollen_rag_sentence"
    assert mocked_stack.dup_args["bucket"] == "soc"
    assert mocked_stack.dup_args["file_hash"] == ingestion.compute_file_hash(sample_file)


def test_ingest_document_records_embedding_meta(mocked_stack, sample_file):
    ingestion.ingest_document(sample_file, strategy="sentence")
    call = mocked_stack.set_meta_calls[0]
    assert call["index_name"] == "ollen_rag_sentence"
    assert call["embedding_provider"] == "watsonx"  # Settings() default
    assert call["embedding_model"] == ingestion.get_settings().watsonx_embedding_model_id
    # Chunking config recorded with only the sentence-strategy knobs (values from settings)
    s = ingestion.get_settings()
    assert call["chunking"] == {"strategy": "sentence", "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap}


def test_ingest_document_records_overridden_embedding_meta(mocked_stack, sample_file):
    ingestion.ingest_document(
        sample_file, strategy="sentence",
        embedding_provider="fastembed", embedding_model="BAAI/bge-large-en-v1.5",
    )
    call = mocked_stack.set_meta_calls[0]
    assert call["embedding_provider"] == "fastembed"
    assert call["embedding_model"] == "BAAI/bge-large-en-v1.5"


def test_ingest_document_records_overridden_chunk_params(mocked_stack, sample_file):
    # Per-request chunk knobs reach the recorded config (and only strategy-relevant ones survive)
    ingestion.ingest_document(
        sample_file, strategy="sentence",
        chunk_params={"chunk_size": 256, "chunk_overlap": 16, "sentence_window_size": 9},
    )
    # sentence_window_size is irrelevant to the sentence strategy and must be dropped
    assert mocked_stack.set_meta_calls[0]["chunking"] == {"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 16}


def test_ingest_document_chunk_param_override_reaches_parser(mocked_stack, sample_file, monkeypatch):
    # The overridden chunk_size must be what the node parser is actually built with
    seen = {}
    real = ingestion.create_node_parser
    def spy(strategy, embed_model=None, llm=None, settings=None, progress_cb=None):
        seen["chunk_size"] = settings.chunk_size
        return real(strategy, embed_model=embed_model, llm=llm, settings=settings, progress_cb=progress_cb)
    monkeypatch.setattr(ingestion, "create_node_parser", spy)
    ingestion.ingest_document(sample_file, strategy="sentence", chunk_params={"chunk_size": 128})
    assert seen["chunk_size"] == 128


def test_ingest_document_rejects_model_mismatch_on_existing_index(mocked_stack, sample_file):
    # One index = one embedding model: a different provider/model into an index with recorded
    # meta must be refused (even when dims would match), so vector spaces never mix.
    mocked_stack.meta = {"embedding_provider": "watsonx", "embedding_model": "ibm/granite-embedding-278m-multilingual"}
    with pytest.raises(ValueError, match="one index holds a single embedding model"):
        ingestion.ingest_document(
            sample_file, strategy="sentence",
            embedding_provider="fastembed", embedding_model="BAAI/bge-large-en-v1.5",
        )


def test_ingest_document_rejects_chunk_config_mismatch_on_existing_index(mocked_stack, sample_file):
    # Same embedding but different chunk size => still refused (one index = one chunking config)
    s = ingestion.get_settings()
    mocked_stack.meta = {
        "embedding_provider": "watsonx", "embedding_model": s.watsonx_embedding_model_id,
        "chunking": {"strategy": "sentence", "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap},
    }
    with pytest.raises(ValueError, match="one index holds a single chunking config"):
        ingestion.ingest_document(sample_file, strategy="sentence", chunk_params={"chunk_size": s.chunk_size + 111})


def test_ingest_document_allows_matching_config_on_existing_index(mocked_stack, sample_file):
    # Same provider/model + same chunking as recorded is allowed (adding docs to an index).
    s = ingestion.get_settings()
    mocked_stack.meta = {
        "embedding_provider": "watsonx", "embedding_model": s.watsonx_embedding_model_id,
        "chunking": {"strategy": "sentence", "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap},
    }
    result = ingestion.ingest_document(sample_file, strategy="sentence")
    assert result["index"] == "ollen_rag_sentence"


def test_ingest_document_duplicate_skip_does_not_record_meta(mocked_stack, sample_file):
    mocked_stack.duplicate = "existing.pdf"
    ingestion.ingest_document(sample_file, strategy="sentence")
    assert mocked_stack.set_meta_calls == []


def test_job_lifecycle(mocked_stack, tmp_path):
    # A job must transition pending -> completed and clean up its temp file
    tmp_file = tmp_path / "doc.pdf"
    tmp_file.write_bytes(b"fake")
    job = ingestion.create_job()
    assert job.status == "pending"
    ingestion.run_ingestion_job(job.job_id, str(tmp_file), "sentence", None, {"team": "soc"}, "doc.pdf")
    assert ingestion.JOBS[job.job_id].status == "completed"
    assert ingestion.JOBS[job.job_id].result["num_nodes"] >= 1
    assert not tmp_file.exists()


class _FakeEnrichLLM:
    """Fake llamaindex LLM for enrichment wiring tests: canned keywords, call counter."""
    def __init__(self):
        self.calls = 0
    def predict(self, prompt, **kwargs):
        self.calls += 1
        return "pandas, dataframe"


def test_ingest_document_enrich_keywords_true(mocked_stack, sample_file, monkeypatch):
    llm = _FakeEnrichLLM()
    monkeypatch.setattr(ingestion, "create_llm", lambda settings=None: llm)
    result = ingestion.ingest_document(sample_file, strategy="sentence", enrich_keywords=True)
    assert result["enriched"] is True
    assert llm.calls >= 1
    assert mocked_stack.added, "pipeline stored no nodes"
    for node in mocked_stack.added:
        assert node.metadata["keywords"] == "pandas, dataframe"
        assert "keywords" in node.excluded_llm_metadata_keys


def test_ingest_document_enrich_default_off_no_llm_call(mocked_stack, sample_file, monkeypatch):
    llm = _FakeEnrichLLM()
    monkeypatch.setattr(ingestion, "create_llm", lambda settings=None: llm)
    result = ingestion.ingest_document(sample_file, strategy="sentence")
    assert result["enriched"] is False
    assert llm.calls == 0


def test_ingest_document_enrich_none_honors_settings_default(mocked_stack, sample_file, monkeypatch):
    from src.settings import Settings
    monkeypatch.setattr(ingestion, "get_settings", lambda: Settings(enrich_keywords=True))
    llm = _FakeEnrichLLM()
    monkeypatch.setattr(ingestion, "create_llm", lambda settings=None: llm)
    result = ingestion.ingest_document(sample_file, strategy="sentence", enrich_keywords=None)
    assert result["enriched"] is True
    assert llm.calls >= 1


def test_ingest_document_explicit_false_overrides_settings_true(mocked_stack, sample_file, monkeypatch):
    from src.settings import Settings
    monkeypatch.setattr(ingestion, "get_settings", lambda: Settings(enrich_keywords=True))
    llm = _FakeEnrichLLM()
    monkeypatch.setattr(ingestion, "create_llm", lambda settings=None: llm)
    result = ingestion.ingest_document(sample_file, strategy="sentence", enrich_keywords=False)
    assert result["enriched"] is False
    assert llm.calls == 0


def test_ingest_document_file_hash_excluded_from_embed_and_llm_modes(mocked_stack, sample_file):
    # sha256 hex is pure noise in embedding text / LLM context — must be excluded even without enrichment
    from llama_index.core.schema import MetadataMode
    result = ingestion.ingest_document(sample_file, strategy="sentence")
    for node in mocked_stack.added:
        assert result["file_hash"] not in node.get_content(metadata_mode=MetadataMode.EMBED)
        assert result["file_hash"] not in node.get_content(metadata_mode=MetadataMode.LLM)
        assert node.metadata["file_hash"] == result["file_hash"]  # still stored for dedup


def test_ingest_document_enrichment_llm_error_fails_job(mocked_stack, sample_file, monkeypatch):
    class _BoomLLM:
        def predict(self, prompt, **kwargs):
            raise RuntimeError("llm down")
    monkeypatch.setattr(ingestion, "create_llm", lambda settings=None: _BoomLLM())
    job = ingestion.create_job()
    ingestion.run_ingestion_job(job.job_id, sample_file, "sentence", None, None, "doc.pdf", True)
    assert ingestion.JOBS[job.job_id].status == "failed"
    assert "llm down" in ingestion.JOBS[job.job_id].detail


def test_job_failure(mocked_stack, monkeypatch, tmp_path):
    # Any exception inside ingestion must mark the job failed with a detail message
    monkeypatch.setattr(ingestion, "LiteParse", _FakeLiteParseEmpty)
    tmp_file = tmp_path / "doc.pdf"
    tmp_file.write_bytes(b"fake")
    job = ingestion.create_job()
    ingestion.run_ingestion_job(job.job_id, str(tmp_file), "sentence", None, None, "doc.pdf")
    assert ingestion.JOBS[job.job_id].status == "failed"
    assert ingestion.JOBS[job.job_id].detail


def test_ingest_document_reports_progress(mocked_stack, sample_file):
    # Percents must be monotonic; stage sequence starts at parsing
    seen = []
    ingestion.ingest_document(sample_file, strategy="sentence", progress_cb=lambda pct, stage: seen.append((pct, stage)))
    assert seen, "no progress reported"
    pcts = [p for p, _ in seen]
    assert pcts == sorted(pcts)
    assert seen[0] == (2, "parsing")
    assert any(s == "chunking" for _, s in seen)


def test_job_finishes_with_progress_100_and_stage_done(mocked_stack, tmp_path):
    tmp_file = tmp_path / "doc.pdf"
    tmp_file.write_bytes(b"fake")
    job = ingestion.create_job()
    assert (job.progress, job.stage) == (0, None)
    ingestion.run_ingestion_job(job.job_id, str(tmp_file), "sentence", None, None, "doc.pdf")
    assert ingestion.JOBS[job.job_id].progress == 100
    assert ingestion.JOBS[job.job_id].stage == "done"


def test_duplicate_skip_also_ends_at_100(mocked_stack, tmp_path):
    mocked_stack.duplicate = "existing.pdf"
    tmp_file = tmp_path / "doc.pdf"
    tmp_file.write_bytes(b"fake")
    job = ingestion.create_job()
    ingestion.run_ingestion_job(job.job_id, str(tmp_file), "sentence", None, None, "doc.pdf")
    assert ingestion.JOBS[job.job_id].status == "completed"
    assert ingestion.JOBS[job.job_id].progress == 100
    assert ingestion.JOBS[job.job_id].stage == "done"