"""Retrieval eval harness: golden YAML datasets scored with hit-rate/recall@k/MRR, per bucket.

Buckets are mandatory on every case (bucket separation is a service invariant) and are
applied as a metadata.bucket == filter on each retrieval call.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from llama_index.core.schema import NodeWithScore
from src.rag.retrieval import retrieve

@dataclass
class ExpectedChunk:
    """One golden target: a source document, optionally narrowed to a chunk substring."""
    file_name: str
    contains: str | None = None

@dataclass
class EvalCase:
    """One golden query with its mandatory bucket scope and expected sources."""
    query: str
    bucket: str
    expected: list[ExpectedChunk]
    strategy: str | None = None
    index_name: str | None = None

@dataclass
class EvalDataset:
    """A parsed golden dataset (see config/eval/example.yaml for the schema)."""
    cases: list[EvalCase]
    description: str = ""

def parse_dataset(data: dict) -> EvalDataset:
    """Validate raw dataset dict (from YAML or the API) and build an EvalDataset.
    Raises ValueError naming the offending case; bucket is mandatory by service invariant."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Eval dataset must contain a non-empty 'cases' list")
    cases = []
    for i, raw in enumerate(data["cases"]):
        if not raw.get("query"):
            raise ValueError(f"case {i}: 'query' is required")
        if not raw.get("bucket"):
            raise ValueError(f"case {i}: 'bucket' is required (bucket separation is mandatory)")
        if not raw.get("expected"):
            raise ValueError(f"case {i}: non-empty 'expected' list is required")
        expected = []
        for e in raw["expected"]:
            # Guard shape here so malformed API payloads surface as ValueError (-> 422), not KeyError (-> 500)
            if not isinstance(e, dict) or not e.get("file_name"):
                raise ValueError(f"case {i}: each expected entry needs 'file_name'")
            expected.append(ExpectedChunk(file_name=e["file_name"], contains=e.get("contains")))
        cases.append(EvalCase(
            query=raw["query"], bucket=raw["bucket"], expected=expected,
            strategy=raw.get("strategy"), index_name=raw.get("index_name"),
        ))
    return EvalDataset(cases=cases, description=data.get("description", ""))

def load_dataset(path: str | Path) -> EvalDataset:
    """Read and validate a golden dataset YAML file."""
    return parse_dataset(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

def _matches(expected: ExpectedChunk, node: NodeWithScore) -> bool:
    """Doc-level match on file_name, optionally refined by a case-insensitive substring."""
    if node.node.metadata.get("file_name") != expected.file_name:
        return False
    return expected.contains is None or expected.contains.lower() in node.node.get_content().lower()

@dataclass
class CaseResult:
    """Per-case outcome: how many expected targets were found and where the first hit ranked."""
    query: str
    bucket: str
    matched: int
    expected: int
    first_rank: int | None
    retrieved: int
    expected_chunks: list[ExpectedChunk] = field(default_factory=list)
    retrieved_nodes: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Fraction of this case's expected targets found in the retrieved set."""
        return self.matched / self.expected

    @property
    def hit(self) -> bool:
        """True when at least one expected target was retrieved."""
        return self.matched > 0

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first matching node; 0 when nothing matched."""
        return 1.0 / self.first_rank if self.first_rank else 0.0

@dataclass
class EvalReport:
    """Aggregated eval outcome: overall and per-bucket hit-rate/recall/MRR plus case rows."""
    results: list[CaseResult]
    params: dict = field(default_factory=dict)

    @staticmethod
    def _aggregate(results: list[CaseResult]) -> dict:
        """Mean hit-rate/recall/MRR over a result subset."""
        n = len(results)
        return {
            "cases": n,
            "hit_rate": sum(r.hit for r in results) / n if n else 0.0,
            "recall": sum(r.recall for r in results) / n if n else 0.0,
            "mrr": sum(r.reciprocal_rank for r in results) / n if n else 0.0,
        }

    def to_dict(self) -> dict:
        """JSON-ready report: overall + per-bucket aggregates + per-case rows."""
        buckets = sorted({r.bucket for r in self.results})
        return {
            "params": self.params,
            "overall": self._aggregate(self.results),
            "per_bucket": {b: self._aggregate([r for r in self.results if r.bucket == b]) for b in buckets},
            "cases": [
                {
                    "query": r.query, "bucket": r.bucket, "matched": r.matched, "expected": r.expected,
                    "first_rank": r.first_rank, "retrieved": r.retrieved,
                    "recall": r.recall, "reciprocal_rank": r.reciprocal_rank,
                    "expected_chunks": [
                        {"file_name": e.file_name, "contains": e.contains}
                        for e in r.expected_chunks
                    ],
                    "retrieved_nodes": r.retrieved_nodes,
                }
                for r in self.results
            ],
        }

    def format_table(self) -> str:
        """Human-readable CLI table: one line per case, then per-bucket and overall aggregates."""
        d = self.to_dict()
        lines = [f"{'bucket':<20} {'matched':>7} {'expected':>8} {'rank':>5}  query"]
        for c in d["cases"]:
            rank = str(c["first_rank"]) if c["first_rank"] else "-"
            lines.append(f"{c['bucket']:<20} {c['matched']:>7} {c['expected']:>8} {rank:>5}  {c['query'][:60]}")
        lines.append("")
        lines.append(f"{'scope':<20} {'cases':>5} {'hit_rate':>8} {'recall':>7} {'mrr':>6}")
        for name, agg in list(d["per_bucket"].items()) + [("OVERALL", d["overall"])]:
            lines.append(f"{name:<20} {agg['cases']:>5} {agg['hit_rate']:>8.2f} {agg['recall']:>7.2f} {agg['mrr']:>6.2f}")
        return "\n".join(lines)

def evaluate(
    dataset: EvalDataset,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    similarity_threshold: float | None = None,
    use_rerank: bool = True,
    index_name: str | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> EvalReport:
    """Run every case through the real retrieval pipeline (bucket-filtered) and score it.

    When index_name is given it overrides each case's own index/strategy, so the whole
    dataset runs against one chosen index (e.g. an index named by embedding model rather
    than by chunking strategy). Otherwise the per-case index_name/strategy is used.

    reranker_provider/reranker_model select the reranker for this run (None = the configured
    default), which is what makes two runs over the same dataset a like-for-like comparison.
    Reported scores are 0-1 relevance probabilities; ranking metrics are scale-invariant."""
    results = []
    for case in dataset.cases:
        nodes = retrieve(
            case.query,
            strategy=None if index_name else case.strategy,
            index_name=index_name or case.index_name,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            raw_filters=[{"key": "bucket", "value": case.bucket, "operator": "=="}],
            similarity_threshold=similarity_threshold,
            use_rerank=use_rerank,
            reranker_provider=reranker_provider,
            reranker_model=reranker_model,
        )
        matched = sum(1 for e in case.expected if any(_matches(e, n) for n in nodes))
        first_rank = next(
            (rank for rank, n in enumerate(nodes, start=1) if any(_matches(e, n) for e in case.expected)),
            None,
        )
        retrieved_nodes = [
            {
                "rank": rank,
                "file_name": n.node.metadata.get("file_name", ""),
                "score": round(float(n.score), 4) if n.score is not None else None,
                "text": n.node.get_content(),  # full chunk; the UI clamps + offers Expand
                "matched": any(_matches(e, n) for e in case.expected),
            }
            for rank, n in enumerate(nodes, start=1)
        ]
        results.append(CaseResult(
            query=case.query, bucket=case.bucket, matched=matched,
            expected=len(case.expected), first_rank=first_rank, retrieved=len(nodes),
            expected_chunks=case.expected, retrieved_nodes=retrieved_nodes,
        ))
    params = {
        "top_k": top_k, "rerank_top_n": rerank_top_n,
        "similarity_threshold": similarity_threshold, "use_rerank": use_rerank,
        "reranker_provider": reranker_provider, "reranker_model": reranker_model,
    }
    return EvalReport(results=results, params=params)

def main() -> None:
    """CLI: python -m src.rag.evaluation --dataset config/eval/golden.yaml [tuning flags]."""
    from src.logger import OllenLogger
    OllenLogger.setup()
    parser = argparse.ArgumentParser(description="Run the retrieval eval harness against a golden dataset")
    parser.add_argument("--dataset", required=True, help="Path to a golden dataset YAML")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Fused-score floor (overrides settings)")
    parser.add_argument("--no-rerank", action="store_true", help="Score the raw fused ranking without reranking")
    parser.add_argument("--index-name", default=None, help="Run all cases against this index, overriding per-case index/strategy")
    parser.add_argument("--reranker-provider", default=None, help="Reranker provider (sentence-transformers | litellm | litellm-watsonx); default: OLLEN_RAG_RERANKER_PROVIDER")
    parser.add_argument("--reranker-model", default=None, help="Model to rerank with (a model id from that provider's list in config/reranker_models.yaml)")
    args = parser.parse_args()
    report = evaluate(
        load_dataset(args.dataset),
        top_k=args.top_k, rerank_top_n=args.rerank_top_n,
        similarity_threshold=args.threshold, use_rerank=not args.no_rerank,
        index_name=args.index_name, reranker_provider=args.reranker_provider,
        reranker_model=args.reranker_model,
    )
    print(report.format_table())

if __name__ == "__main__":
    main()
