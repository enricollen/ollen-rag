"""Tests for the retrieval eval harness: dataset loading/validation, metric math, aggregation."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from src.rag import evaluation as eval_mod

def _node(file_name: str, text: str = "some chunk text", score: float = 0.5) -> NodeWithScore:
    """Retrieved-node stand-in carrying the file_name metadata the matcher keys on."""
    return NodeWithScore(node=TextNode(text=text, metadata={"file_name": file_name}), score=score)

_VALID = {
    "description": "d",
    "cases": [
        {"query": "q1", "bucket": "b1", "expected": [{"file_name": "a.pdf"}]},
        {"query": "q2", "bucket": "b2", "expected": [{"file_name": "b.pdf", "contains": "Needle"}, {"file_name": "c.pdf"}]},
    ],
}

def test_parse_dataset_valid():
    ds = eval_mod.parse_dataset(_VALID)
    assert len(ds.cases) == 2
    assert ds.cases[1].expected[0].contains == "Needle"

def test_parse_dataset_requires_bucket():
    bad = {"cases": [{"query": "q", "expected": [{"file_name": "a.pdf"}]}]}
    with pytest.raises(ValueError, match="bucket"):
        eval_mod.parse_dataset(bad)

def test_parse_dataset_requires_query_and_expected():
    with pytest.raises(ValueError, match="query"):
        eval_mod.parse_dataset({"cases": [{"bucket": "b", "expected": [{"file_name": "a"}]}]})
    with pytest.raises(ValueError, match="expected"):
        eval_mod.parse_dataset({"cases": [{"query": "q", "bucket": "b"}]})

def test_parse_dataset_requires_expected_file_name():
    # malformed expected entries (missing file_name / non-dict) must fail as ValueError, not KeyError
    with pytest.raises(ValueError, match="file_name"):
        eval_mod.parse_dataset({"cases": [{"query": "q", "bucket": "b", "expected": [{"contains": "x"}]}]})
    with pytest.raises(ValueError, match="file_name"):
        eval_mod.parse_dataset({"cases": [{"query": "q", "bucket": "b", "expected": ["a.pdf"]}]})

def test_load_dataset_reads_yaml(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text("cases:\n  - query: q\n    bucket: b\n    expected:\n      - file_name: a.pdf\n", encoding="utf-8")
    assert len(eval_mod.load_dataset(p).cases) == 1

def test_evaluate_metrics(monkeypatch):
    """q1: expected a.pdf found at rank 2 -> hit, recall 1, rr 0.5.
    q2: expects b.pdf(contains Needle) + c.pdf; only c.pdf found at rank 1 -> hit, recall 0.5, rr 1.0."""
    returns = {
        "q1": [_node("x.pdf"), _node("a.pdf")],
        "q2": [_node("c.pdf"), _node("b.pdf", text="no match here")],
    }
    captured_filters = []
    def _fake_retrieve(query, **kwargs):
        captured_filters.append(kwargs["raw_filters"])
        return returns[query]
    monkeypatch.setattr(eval_mod, "retrieve", _fake_retrieve)
    report = eval_mod.evaluate(eval_mod.parse_dataset(_VALID))
    overall = report.to_dict()["overall"]
    assert overall["cases"] == 2
    assert overall["hit_rate"] == 1.0
    assert overall["recall"] == pytest.approx(0.75)
    assert overall["mrr"] == pytest.approx(0.75)
    # Bucket invariant: every retrieve call carries the case's bucket as an == filter
    assert captured_filters[0] == [{"key": "bucket", "value": "b1", "operator": "=="}]
    assert captured_filters[1] == [{"key": "bucket", "value": "b2", "operator": "=="}]

def test_evaluate_forwards_reranker_model(monkeypatch):
    """Comparing two rerankers on one golden dataset is the harness's reason to exist,
    so evaluate() must pass reranker_model down to retrieve() unchanged."""
    captured = []
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kwargs: captured.append(kwargs.get("reranker_model")) or [])
    eval_mod.evaluate(eval_mod.parse_dataset(_VALID), reranker_model="BAAI/bge-reranker-v2-m3")
    assert captured == ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3"]

def test_evaluate_reranker_model_recorded_in_params(monkeypatch):
    """The report must name the reranker it used, or two runs are indistinguishable."""
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kwargs: [])
    report = eval_mod.evaluate(eval_mod.parse_dataset(_VALID), reranker_model="models/reranker")
    assert report.to_dict()["params"]["reranker_model"] == "models/reranker"

def test_evaluate_per_bucket_split(monkeypatch):
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kwargs: [_node("a.pdf")] if query == "q1" else [])
    report = eval_mod.evaluate(eval_mod.parse_dataset(_VALID))
    per_bucket = report.to_dict()["per_bucket"]
    assert per_bucket["b1"]["hit_rate"] == 1.0
    assert per_bucket["b2"]["hit_rate"] == 0.0

def test_format_table_smoke(monkeypatch):
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kwargs: [])
    table = eval_mod.evaluate(eval_mod.parse_dataset(_VALID)).format_table()
    assert "hit_rate" in table and "b1" in table
