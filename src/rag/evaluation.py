"""Retrieval eval harness: golden YAML datasets scored with hit-rate/recall@k/MRR, per bucket.

Buckets are mandatory on every case (bucket separation is a service invariant) and are
applied as a metadata.bucket == filter on each retrieval call.
"""
import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml
from llama_index.core.schema import NodeWithScore
from src.factories.vector_store import build_index_name, create_backend
from src.logger import OllenLogger, preview
from src.rag.retrieval import retrieve, retrieve_debug
from src.settings import get_settings

# Progress logger: an eval walks the whole dataset one case at a time, so it narrates each stage
# (start → per-case hit/miss → aggregation → done) to make a long run legible in the console.
logger = OllenLogger("eval")

# Rank cutoffs reported as recall@k / precision@k / nDCG@k curves. PRIMARY_K is the single cutoff
# the scalar "ndcg" summarizes. BOOTSTRAP_ITERS/SEED fix the resampling so a run is reproducible.
CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)
PRIMARY_K: int = 10
BOOTSTRAP_ITERS: int = 1000
BOOTSTRAP_SEED: int = 0

def _dcg(rels: list[int], k: int) -> float:
    """Discounted cumulative gain over the first k relevance grades (rank 1 divisor = log2(2) = 1)."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

def _ndcg_at(rels: list[int], k: int) -> float:
    """nDCG@k: DCG of the actual ranking over the ideal (all relevant sorted to the top). 0 when no ideal gain."""
    ideal = _dcg(sorted(rels, reverse=True), k)
    return _dcg(rels, k) / ideal if ideal else 0.0

def _average_precision(rels: list[int]) -> float:
    """AP: mean of precision@rank taken at every relevant position; 0 when nothing relevant retrieved.
    Denominator is relevant-retrieved count, so the value stays in [0, 1] even when several chunks
    match the same expected source."""
    hits = 0
    total = 0.0
    for rank, rel in enumerate(rels, start=1):
        if rel:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0

def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of *values* (0.0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]

def _bootstrap_ci(per_case: list[float]) -> list[float]:
    """95% bootstrap confidence interval for the mean of a per-case metric.
    Resamples cases with replacement (fixed seed -> reproducible); [0,0] for an empty input."""
    if not per_case:
        return [0.0, 0.0]
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(per_case)
    means = sorted(
        sum(per_case[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_ITERS)
    )
    lo = means[int(0.025 * BOOTSTRAP_ITERS)]
    hi = means[int(0.975 * BOOTSTRAP_ITERS)]
    return [round(lo, 4), round(hi, 4)]

@dataclass
class ExpectedChunk:
    """One golden target: a source document, optionally narrowed to a chunk substring.

    relevance is the graded gain (>=1) used by nDCG: a target marked relevance 3 contributes
    more than one marked 1 when ranked high. Binary metrics (hit/recall/precision/MAP) treat any
    relevance >= 1 as simply 'relevant', so a dataset that omits grades behaves exactly as before."""
    file_name: str
    contains: str | None = None
    relevance: int = 1

@dataclass
class EvalCase:
    """One golden query with its expected sources. Eval is bucket-agnostic: `bucket` is an
    optional reporting label for slicing metrics, never a retrieval filter."""
    query: str
    expected: list[ExpectedChunk]
    bucket: str | None = None
    strategy: str | None = None
    index_name: str | None = None

@dataclass
class EvalDataset:
    """A parsed golden dataset (see config/eval/example.yaml for the schema)."""
    cases: list[EvalCase]
    description: str = ""

def parse_dataset(data: dict) -> EvalDataset:
    """Validate raw dataset dict (from YAML or the API) and build an EvalDataset.
    Raises ValueError naming the offending case. `bucket` is optional — eval is bucket-agnostic,
    so a case's bucket (if given) only labels its metrics, it never scopes retrieval."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Eval dataset must contain a non-empty 'cases' list")
    cases = []
    for i, raw in enumerate(data["cases"]):
        if not raw.get("query"):
            raise ValueError(f"case {i}: 'query' is required")
        if not raw.get("expected"):
            raise ValueError(f"case {i}: non-empty 'expected' list is required")
        expected = []
        for e in raw["expected"]:
            # Guard shape here so malformed API payloads surface as ValueError (-> 422), not KeyError (-> 500)
            if not isinstance(e, dict) or not e.get("file_name"):
                raise ValueError(f"case {i}: each expected entry needs 'file_name'")
            # relevance is optional; when present it must be a positive int grade (bool excluded)
            relevance = e.get("relevance", 1)
            if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 1:
                raise ValueError(f"case {i}: 'relevance' must be a positive integer (got {relevance!r})")
            expected.append(ExpectedChunk(file_name=e["file_name"], contains=e.get("contains"), relevance=relevance))
        cases.append(EvalCase(
            query=raw["query"], bucket=raw.get("bucket"), expected=expected,
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

def _grade(expected: list[ExpectedChunk], node: NodeWithScore) -> int:
    """Graded relevance of *node*: the highest grade among the expected targets it matches,
    or 0 if it matches none. Feeds nDCG; binary relevance is just grade > 0."""
    return max((e.relevance for e in expected if _matches(e, node)), default=0)

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
    # Binary relevance per retrieved rank (1 = node matched an expected target); feeds the
    # binary rank-aware metrics (precision/AP). Empty when nothing was retrieved.
    rank_relevance: list[int] = field(default_factory=list)
    # Graded gain per retrieved rank (the matched target's relevance grade); feeds nDCG.
    # Equals rank_relevance when the dataset uses no explicit grades.
    rank_gains: list[int] = field(default_factory=list)
    # First rank at which each expected target appeared, or None if never; drives recall@k.
    target_ranks: list[int | None] = field(default_factory=list)
    # End-to-end retrieval latency for this case (ms), including embed + rerank.
    latency_ms: float = 0.0

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

    def precision_at(self, k: int) -> float:
        """Fraction of the top-k retrieved nodes that are relevant (divided by k, so missing
        results below the cutoff count against precision)."""
        return sum(self.rank_relevance[:k]) / k

    def recall_at(self, k: int) -> float:
        """Fraction of expected targets whose first matching node ranks within the top k."""
        found = sum(1 for r in self.target_ranks if r is not None and r <= k)
        return found / len(self.target_ranks) if self.target_ranks else 0.0

    def ndcg_at(self, k: int) -> float:
        """nDCG@k over the graded per-rank gains (falls back to binary when no grades given)."""
        return _ndcg_at(self.rank_gains or self.rank_relevance, k)

    @property
    def average_precision(self) -> float:
        """Average precision over the full retrieved ranking."""
        return _average_precision(self.rank_relevance)

@dataclass
class EvalReport:
    """Aggregated eval outcome: overall and per-bucket hit-rate/recall/MRR plus case rows."""
    results: list[CaseResult]
    params: dict = field(default_factory=dict)

    @staticmethod
    def _aggregate(results: list[CaseResult]) -> dict:
        """Mean hit-rate/recall/MRR/MAP/nDCG over a result subset, plus recall/precision/nDCG
        curves at CUTOFFS, latency percentiles, and 95% bootstrap CIs on the scalar metrics."""
        n = len(results)
        if not n:
            return {
                "cases": 0, "hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "map": 0.0, "ndcg": 0.0,
                "recall_at": {}, "precision_at": {}, "ndcg_at": {},
                "latency_ms": {"p50": 0.0, "p95": 0.0, "mean": 0.0}, "ci": {},
            }

        def mean(values: list[float]) -> float:
            return sum(values) / n

        # Per-case metric columns, reused for both the means and the bootstrap CIs
        hit = [float(r.hit) for r in results]
        recall = [r.recall for r in results]
        rr = [r.reciprocal_rank for r in results]
        ap = [r.average_precision for r in results]
        ndcg = [r.ndcg_at(PRIMARY_K) for r in results]
        latency = [r.latency_ms for r in results]
        return {
            "cases": n,
            "hit_rate": mean(hit),
            "recall": mean(recall),
            "mrr": mean(rr),
            "map": mean(ap),
            "ndcg": mean(ndcg),
            # String keys so the JSON payload and the UI agree (JSON has no int keys anyway)
            "recall_at": {str(k): mean([r.recall_at(k) for r in results]) for k in CUTOFFS},
            "precision_at": {str(k): mean([r.precision_at(k) for r in results]) for k in CUTOFFS},
            "ndcg_at": {str(k): mean([r.ndcg_at(k) for r in results]) for k in CUTOFFS},
            "latency_ms": {
                "p50": _percentile(latency, 50),
                "p95": _percentile(latency, 95),
                "mean": mean(latency),
            },
            "ci": {
                "hit_rate": _bootstrap_ci(hit),
                "recall": _bootstrap_ci(recall),
                "mrr": _bootstrap_ci(rr),
                "map": _bootstrap_ci(ap),
                "ndcg": _bootstrap_ci(ndcg),
            },
        }

    def to_dict(self) -> dict:
        """JSON-ready report: overall + per-bucket aggregates + per-case rows."""
        buckets = sorted({r.bucket for r in self.results if r.bucket})  # only labelled cases; agnostic runs have none
        return {
            "params": self.params,
            "overall": self._aggregate(self.results),
            "per_bucket": {b: self._aggregate([r for r in self.results if r.bucket == b]) for b in buckets},
            "cases": [
                {
                    "query": r.query, "bucket": r.bucket, "matched": r.matched, "expected": r.expected,
                    "first_rank": r.first_rank, "retrieved": r.retrieved,
                    "recall": r.recall, "reciprocal_rank": r.reciprocal_rank,
                    "average_precision": r.average_precision, "ndcg": r.ndcg_at(PRIMARY_K),
                    "latency_ms": round(r.latency_ms, 1),
                    "recall_at": {str(k): r.recall_at(k) for k in CUTOFFS},
                    "precision_at": {str(k): r.precision_at(k) for k in CUTOFFS},
                    "ndcg_at": {str(k): r.ndcg_at(k) for k in CUTOFFS},
                    "expected_chunks": [
                        {"file_name": e.file_name, "contains": e.contains, "relevance": e.relevance}
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
            lines.append(f"{(c['bucket'] or '-'):<20} {c['matched']:>7} {c['expected']:>8} {rank:>5}  {c['query'][:60]}")
        lines.append("")
        lines.append(
            f"{'scope':<20} {'cases':>5} {'hit_rate':>8} {'recall':>7} {'mrr':>6} "
            f"{'ndcg':>6} {'map':>6} {'p50ms':>7}"
        )
        for name, agg in list(d["per_bucket"].items()) + [("OVERALL", d["overall"])]:
            lines.append(
                f"{name:<20} {agg['cases']:>5} {agg['hit_rate']:>8.2f} {agg['recall']:>7.2f} "
                f"{agg['mrr']:>6.2f} {agg['ndcg']:>6.2f} {agg['map']:>6.2f} "
                f"{agg['latency_ms']['p50']:>7.1f}"
            )
        return "\n".join(lines)

def _score_case(case: EvalCase, nodes: list[NodeWithScore], latency_ms: float) -> CaseResult:
    """Score one retrieved node list against a case's expected targets into a CaseResult.
    Shared by the whole-pipeline eval and the per-leg attribution eval so both compute metrics
    identically."""
    # First rank each expected target appears at (None = never); drives matched/first_rank/recall@k
    target_ranks = [
        next((rank for rank, n in enumerate(nodes, start=1) if _matches(e, n)), None)
        for e in case.expected
    ]
    matched = sum(1 for r in target_ranks if r is not None)
    first_rank = min((r for r in target_ranks if r is not None), default=None)
    # Graded gains per rank (for nDCG) and their binary projection (for precision/AP)
    rank_gains = [_grade(case.expected, n) for n in nodes]
    rank_relevance = [1 if g > 0 else 0 for g in rank_gains]
    retrieved_nodes = [
        {
            "rank": rank,
            "file_name": n.node.metadata.get("file_name", ""),
            "score": round(float(n.score), 4) if n.score is not None else None,
            "text": n.node.get_content(),  # full chunk; the UI clamps + offers Expand
            "matched": rank_gains[rank - 1] > 0,
        }
        for rank, n in enumerate(nodes, start=1)
    ]
    return CaseResult(
        query=case.query, bucket=case.bucket, matched=matched,
        expected=len(case.expected), first_rank=first_rank, retrieved=len(nodes),
        expected_chunks=case.expected, retrieved_nodes=retrieved_nodes,
        rank_relevance=rank_relevance, rank_gains=rank_gains, target_ranks=target_ranks,
        latency_ms=latency_ms,
    )

def _resolve_system(dataset: EvalDataset, index_name: str | None, settings) -> dict:
    """Snapshot of the backing setup a run actually hit: active vector store, plus the recorded
    build config (embedding provider/model, chunking) of every index the dataset's cases resolved
    to. Same index-resolution rule as retrieve(): an explicit index_name overrides every case;
    otherwise each case's own strategy/index_name is used, so a single run can span >1 index.
    Recorded so a later A/B compare shows which system produced each run, not just its metrics."""
    backend = create_backend(settings)
    targets = {
        build_index_name(None if index_name else case.strategy, index_name or case.index_name, settings)
        for case in dataset.cases
    }
    indices = {}
    for target in sorted(targets):
        meta = backend.get_index_meta(target) or {}
        indices[target] = {
            "embedding_provider": meta.get("embedding_provider"),
            "embedding_model": meta.get("embedding_model"),
            "chunking": meta.get("chunking"),
        }
    return {"vector_store": settings.vector_store, "indices": indices}

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
    total = len(dataset.cases)
    logger.info("▶ eval starting · %d cases · index=%s · rerank=%s",
                total, index_name or "per-case", "on" if use_rerank else "off")
    results = []
    for i, case in enumerate(dataset.cases, start=1):
        started = time.perf_counter()
        nodes = retrieve(
            case.query,
            strategy=None if index_name else case.strategy,
            index_name=index_name or case.index_name,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            raw_filters=None,  # bucket-agnostic: search the whole index; the user matches dataset↔index
            similarity_threshold=similarity_threshold,
            use_rerank=use_rerank,
            reranker_provider=reranker_provider,
            reranker_model=reranker_model,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = _score_case(case, nodes, latency_ms)
        results.append(result)
        logger.info("  [%d/%d] %-4s %d/%d matched · %4.0fms · %s",
                    i, total, "HIT" if result.hit else "MISS",
                    result.matched, result.expected, latency_ms, preview(case.query, 60))
    hits = sum(1 for r in results if r.hit)
    mean_latency = sum(r.latency_ms for r in results) / total if total else 0.0
    logger.info("■ eval done · %d/%d hits · mean %.0fms · aggregating metrics…",
                hits, total, mean_latency)
    settings = get_settings()
    params = {
        "top_k": top_k, "rerank_top_n": rerank_top_n,
        "similarity_threshold": similarity_threshold, "use_rerank": use_rerank,
        "reranker_provider": reranker_provider, "reranker_model": reranker_model,
        "system": _resolve_system(dataset, index_name, settings),
    }
    return EvalReport(results=results, params=params)

# Retrieval legs scored separately by evaluate_legs (mirrors retrieve_debug's keys)
LEGS: tuple[str, ...] = ("bm25", "dense", "hybrid", "reranked")

def evaluate_legs(
    dataset: EvalDataset,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    similarity_threshold: float | None = None,
    index_name: str | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> dict:
    """Score each retrieval leg (bm25/dense/hybrid/reranked) separately, exposing what fusion and
    the cross-encoder contribute. Returns per-leg aggregates plus rerank_lift = reranked - hybrid
    on the scalar metrics. Per-leg latency is not measured (one retrieve_debug call serves every
    leg), so latency fields read 0 here."""
    total = len(dataset.cases)
    logger.info("▶ per-leg eval starting · %d cases · index=%s · legs=%s",
                total, index_name or "per-case", "/".join(LEGS))
    per_leg_results: dict[str, list[CaseResult]] = {leg: [] for leg in LEGS}
    for i, case in enumerate(dataset.cases, start=1):
        debug = retrieve_debug(
            case.query,
            strategy=None if index_name else case.strategy,
            index_name=index_name or case.index_name,
            top_k=top_k, rerank_top_n=rerank_top_n, raw_filters=None,
            similarity_threshold=similarity_threshold,
            reranker_provider=reranker_provider, reranker_model=reranker_model,
        )
        for leg in LEGS:
            per_leg_results[leg].append(_score_case(case, debug.get(leg, []), 0.0))
        logger.info("  [%d/%d] scored all legs · %s", i, total, preview(case.query, 60))
    logger.info("■ per-leg eval done · aggregating metrics…")
    per_leg = {leg: EvalReport(res).to_dict() for leg, res in per_leg_results.items()}
    rerank_lift = {
        m: round(per_leg["reranked"]["overall"][m] - per_leg["hybrid"]["overall"][m], 4)
        for m in ("hit_rate", "recall", "mrr", "ndcg", "map")
    }
    return {
        "per_leg": per_leg,
        "rerank_lift": rerank_lift,
        "params": {
            "top_k": top_k, "rerank_top_n": rerank_top_n,
            "similarity_threshold": similarity_threshold,
            "reranker_provider": reranker_provider, "reranker_model": reranker_model,
        },
    }

def _runs_dir() -> Path:
    """Directory holding saved eval runs (created on demand, under the eval dir)."""
    directory = Path(get_settings().eval_dir) / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_run(report: dict, label: str | None = None, dataset: str | None = None) -> str:
    """Persist a report dict (evaluate(...).to_dict()) under a timestamp id; return the id.
    The record keeps id/timestamp/label/dataset/params/overall at top level for cheap listing,
    with the full report nested under 'report'."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%dT%H%M%S_") + f"{now.microsecond // 1000:03d}"
    payload = {
        "id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "label": label,
        "dataset": dataset,
        "params": report.get("params"),
        "overall": report.get("overall"),
        "report": report,
    }
    (_runs_dir() / f"{run_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_id

def list_runs() -> list[dict]:
    """Saved runs newest-first, without the heavy nested report (id/timestamp/label/params/overall)."""
    runs = []
    for path in sorted(_runs_dir().glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append({key: record.get(key) for key in ("id", "timestamp", "label", "dataset", "params", "overall")})
    return runs

def load_run(run_id: str) -> dict | None:
    """Full saved record for a run id, or None if absent/unreadable."""
    path = _runs_dir() / f"{run_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

# Per-case metric extractors, keyed to the case-row fields to_dict emits; drive paired comparison
_CASE_METRIC = {
    "hit_rate": lambda c: 1.0 if c.get("matched", 0) > 0 else 0.0,
    "recall": lambda c: c.get("recall", 0.0),
    "mrr": lambda c: c.get("reciprocal_rank", 0.0),
    "ndcg": lambda c: c.get("ndcg", 0.0),
    "map": lambda c: c.get("average_precision", 0.0),
}

def _paired_delta_ci(deltas: list[float]) -> dict:
    """Mean of per-case deltas with a 95% bootstrap CI; 'significant' when the CI excludes 0."""
    if not deltas:
        return {"delta": 0.0, "ci": [0.0, 0.0], "significant": False, "n": 0}
    mean_delta = sum(deltas) / len(deltas)
    ci = _bootstrap_ci(deltas)
    return {"delta": round(mean_delta, 4), "ci": ci, "significant": ci[0] > 0 or ci[1] < 0, "n": len(deltas)}

def compare_runs(run_a: dict, run_b: dict) -> dict:
    """Paired A/B comparison of two reports, matching cases by query. Reports mean metric delta
    (B - A) with a bootstrap CI and significance per metric, over the cases both runs share.
    Accepts either a raw report dict or a saved record (with the report nested under 'report')."""
    a = run_a.get("report", run_a)
    b = run_b.get("report", run_b)
    a_cases = {c["query"]: c for c in a.get("cases", [])}
    b_cases = {c["query"]: c for c in b.get("cases", [])}
    common = [q for q in a_cases if q in b_cases]
    metrics = {
        name: _paired_delta_ci([fn(b_cases[q]) - fn(a_cases[q]) for q in common])
        for name, fn in _CASE_METRIC.items()
    }
    return {
        "n_paired": len(common),
        "a_only": [q for q in a_cases if q not in b_cases],
        "b_only": [q for q in b_cases if q not in a_cases],
        "metrics": metrics,
    }

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
    parser.add_argument("--per-leg", action="store_true", help="Score bm25/dense/hybrid/reranked legs separately and print the rerank lift")
    parser.add_argument("--save", action="store_true", help="Persist this run under config/eval/runs for later A/B comparison")
    parser.add_argument("--label", default=None, help="Optional label stored with a --save'd run")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None, help="Compare two saved run ids and exit (paired delta + significance)")
    args = parser.parse_args()
    # Compare mode short-circuits: it reads two saved runs, prints deltas, and exits
    if args.compare:
        a, b = (load_run(rid) for rid in args.compare)
        if a is None or b is None:
            raise SystemExit(f"Unknown run id(s): {args.compare}")
        cmp = compare_runs(a, b)
        print(f"paired cases: {cmp['n_paired']}")
        print(f"{'metric':<10} {'delta':>8} {'95% CI':>18} {'sig':>4}")
        for name, m in cmp["metrics"].items():
            ci = f"[{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]"
            sig = "*" if m["significant"] else ""
            print(f"{name:<10} {m['delta']:>+8.4f} {ci:>18} {sig:>4}")
        return
    dataset = load_dataset(args.dataset)
    if args.per_leg:
        legs = evaluate_legs(
            dataset, top_k=args.top_k, rerank_top_n=args.rerank_top_n,
            similarity_threshold=args.threshold, index_name=args.index_name,
            reranker_provider=args.reranker_provider, reranker_model=args.reranker_model,
        )
        for leg, rep in legs["per_leg"].items():
            ov = rep["overall"]
            print(f"[{leg:<9}] ndcg={ov['ndcg']:.3f} recall={ov['recall']:.3f} mrr={ov['mrr']:.3f} map={ov['map']:.3f}")
        print("rerank lift (reranked - hybrid): " + ", ".join(f"{k}={v:+.4f}" for k, v in legs["rerank_lift"].items()))
        return
    report = evaluate(
        dataset,
        top_k=args.top_k, rerank_top_n=args.rerank_top_n,
        similarity_threshold=args.threshold, use_rerank=not args.no_rerank,
        index_name=args.index_name, reranker_provider=args.reranker_provider,
        reranker_model=args.reranker_model,
    )
    print(report.format_table())
    if args.save:
        run_id = save_run(report.to_dict(), label=args.label, dataset=Path(args.dataset).stem)
        print(f"\nsaved run: {run_id}")

if __name__ == "__main__":
    main()
