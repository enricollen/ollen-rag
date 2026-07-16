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

def test_parse_dataset_bucket_optional():
    # Eval is bucket-agnostic: a case without a bucket parses fine, bucket defaults to None.
    ds = eval_mod.parse_dataset({"cases": [{"query": "q", "expected": [{"file_name": "a.pdf"}]}]})
    assert ds.cases[0].bucket is None

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
    # Bucket-agnostic: eval never applies a bucket (or any) filter — it searches the whole index.
    assert captured_filters == [None, None]

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
    # Tier-1 additions must surface in the CLI table too
    assert "ndcg" in table and "map" in table

# --- Tier 1: ranking-metric math (pure helpers) ---

def test_ndcg_helper():
    import math
    # Perfect ordering (all relevant first) -> nDCG 1.0
    assert eval_mod._ndcg_at([1, 1, 0, 0], 4) == pytest.approx(1.0)
    # Single relevant at rank 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1
    assert eval_mod._ndcg_at([0, 1], 2) == pytest.approx(1 / math.log2(3))
    # No relevant retrieved -> 0
    assert eval_mod._ndcg_at([0, 0], 5) == 0.0

def test_average_precision_helper():
    # Relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert eval_mod._average_precision([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert eval_mod._average_precision([0, 0]) == 0.0

def test_percentile_and_bootstrap():
    assert eval_mod._percentile([10, 20, 30, 40], 50) in (20, 30)
    assert eval_mod._percentile([], 95) == 0.0
    # Degenerate sample: CI collapses to the point value
    assert eval_mod._bootstrap_ci([1.0, 1.0, 1.0]) == [1.0, 1.0]
    lo, hi = eval_mod._bootstrap_ci([0.0, 1.0, 0.0, 1.0])
    assert 0.0 <= lo <= hi <= 1.0

def test_evaluate_extended_metrics(monkeypatch):
    """q1: a.pdf at rank 2. q2: c.pdf at rank 1, b.pdf(Needle) missing."""
    returns = {
        "q1": [_node("x.pdf"), _node("a.pdf")],
        "q2": [_node("c.pdf"), _node("b.pdf", text="no match here")],
    }
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kwargs: returns[query])
    ov = eval_mod.evaluate(eval_mod.parse_dataset(_VALID), top_k=5).to_dict()["overall"]
    # New aggregate scalars + curves + latency + CI all present
    assert {"ndcg", "map"} <= ov.keys()
    assert set(ov["recall_at"]) == {"1", "3", "5", "10"}
    assert set(ov["precision_at"]) == {"1", "3", "5", "10"}
    assert "p50" in ov["latency_ms"] and "p95" in ov["latency_ms"]
    assert len(ov["ci"]["hit_rate"]) == 2 and ov["ci"]["hit_rate"][0] <= ov["ci"]["hit_rate"][1]
    # recall@1: q1=0 (a.pdf at rank 2), q2=0.5 (c.pdf at rank 1 of 2 expected) -> mean 0.25.
    # recall@5: q1=1.0, q2=0.5 -> mean 0.75. Curve rises with k.
    assert ov["recall_at"]["1"] == pytest.approx(0.25)
    assert ov["recall_at"]["5"] == pytest.approx(0.75)

def test_evaluate_case_has_extended_fields(monkeypatch):
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kwargs: [_node("a.pdf")])
    c0 = eval_mod.evaluate(eval_mod.parse_dataset(_VALID), top_k=5).to_dict()["cases"][0]
    assert "average_precision" in c0 and "ndcg" in c0
    assert "latency_ms" in c0
    assert set(c0["precision_at"]) == {"1", "3", "5", "10"}

# --- Tier 2: graded relevance ---

def test_parse_dataset_relevance_default_and_explicit():
    ds = eval_mod.parse_dataset({"cases": [
        {"query": "q", "expected": [{"file_name": "a.pdf"}, {"file_name": "b.pdf", "relevance": 3}]},
    ]})
    assert ds.cases[0].expected[0].relevance == 1  # default
    assert ds.cases[0].expected[1].relevance == 3

def test_parse_dataset_rejects_bad_relevance():
    with pytest.raises(ValueError, match="relevance"):
        eval_mod.parse_dataset({"cases": [{"query": "q", "expected": [{"file_name": "a", "relevance": 0}]}]})
    with pytest.raises(ValueError, match="relevance"):
        eval_mod.parse_dataset({"cases": [{"query": "q", "expected": [{"file_name": "a", "relevance": "hi"}]}]})

def test_graded_relevance_lifts_ndcg(monkeypatch):
    """A high-grade target ranked first should score nDCG higher than a low-grade one first."""
    ds = eval_mod.parse_dataset({"cases": [
        {"query": "q", "expected": [{"file_name": "hi.pdf", "relevance": 3}, {"file_name": "lo.pdf", "relevance": 1}]},
    ]})
    # hi (grade 3) at rank 1, lo (grade 1) at rank 2 -> near-ideal ordering
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kw: [_node("hi.pdf"), _node("lo.pdf")])
    good = eval_mod.evaluate(ds, top_k=5).to_dict()["cases"][0]["ndcg"]
    # lo (grade 1) at rank 1, hi (grade 3) at rank 2 -> worse ordering by gain
    monkeypatch.setattr(eval_mod, "retrieve", lambda query, **kw: [_node("lo.pdf"), _node("hi.pdf")])
    bad = eval_mod.evaluate(ds, top_k=5).to_dict()["cases"][0]["ndcg"]
    assert good == pytest.approx(1.0)
    assert bad < good

# --- Tier 2: per-leg attribution ---

def test_evaluate_legs_reports_lift(monkeypatch):
    debug = {
        "bm25": [_node("x.pdf"), _node("a.pdf")],
        "dense": [_node("a.pdf")],
        "hybrid": [_node("x.pdf"), _node("a.pdf")],   # a at rank 2
        "reranked": [_node("a.pdf"), _node("x.pdf")], # a promoted to rank 1
    }
    monkeypatch.setattr(eval_mod, "retrieve_debug", lambda query, **kw: debug)
    out = eval_mod.evaluate_legs(eval_mod.parse_dataset(_VALID), top_k=5)
    assert set(out["per_leg"]) == {"bm25", "dense", "hybrid", "reranked"}
    # rerank promoted the match -> non-negative nDCG lift over hybrid
    assert out["rerank_lift"]["ndcg"] >= 0.0

# --- Tier 2: run persistence + paired compare ---

def test_save_list_load_run(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_mod, "_runs_dir", lambda: tmp_path)
    report = {"params": {"top_k": 5}, "overall": {"ndcg": 0.5}, "cases": []}
    rid = eval_mod.save_run(report, label="baseline", dataset="golden")
    listed = eval_mod.list_runs()
    assert any(r["id"] == rid and r["label"] == "baseline" for r in listed)
    loaded = eval_mod.load_run(rid)
    assert loaded["report"]["overall"]["ndcg"] == 0.5
    assert eval_mod.load_run("nope") is None

def test_compare_runs_paired_delta():
    def report(ndcgs):
        return {"cases": [
            {"query": f"q{i}", "matched": 1, "recall": 1.0, "reciprocal_rank": 1.0,
             "average_precision": 1.0, "ndcg": v} for i, v in enumerate(ndcgs)
        ]}
    a = report([0.2, 0.2, 0.2, 0.2])
    b = report([0.9, 0.9, 0.9, 0.9])  # uniformly better
    cmp = eval_mod.compare_runs(a, b)
    assert cmp["n_paired"] == 4
    assert cmp["metrics"]["ndcg"]["delta"] == pytest.approx(0.7)
    assert cmp["metrics"]["ndcg"]["significant"] is True
    # Identical runs -> zero delta, not significant
    same = eval_mod.compare_runs(a, a)
    assert same["metrics"]["ndcg"]["delta"] == 0.0
    assert same["metrics"]["ndcg"]["significant"] is False
