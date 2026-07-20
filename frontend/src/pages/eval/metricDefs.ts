// Plain-language glossary for every metric on the page, ported from ui/pages/eval.js's METRIC_DEFS.
export interface MetricDef {
  key: string
  name: string
  short: string
  long: string
  example: string
}

export const METRIC_DEFS: MetricDef[] = [
  {
    key: 'hit_rate',
    name: 'Hit-rate',
    short: 'Did any correct source show up at all?',
    long: 'Share of queries where at least one relevant document lands anywhere in the top-k results.',
    example: '10 queries; 8 had a correct source somewhere in their top-5 → hit-rate 80%.',
  },
  {
    key: 'recall',
    name: 'Recall',
    short: 'How many of the correct sources did we find?',
    long: 'Of all documents that should have been retrieved for a query, the fraction actually returned.',
    example: 'A query has 4 relevant docs; retrieval returned 3 of them → recall 0.75.',
  },
  {
    key: 'precision',
    name: 'Precision@k',
    short: 'How clean are the top-k results?',
    long: 'Of the top-k returned results, the fraction that are actually relevant (the rest are noise).',
    example: 'Top-5 results, 2 relevant + 3 irrelevant → precision@5 = 0.40.',
  },
  {
    key: 'mrr',
    name: 'MRR',
    short: 'How high up was the first correct hit?',
    long: 'Mean Reciprocal Rank — average of 1 / (rank of the first relevant result) across queries. Higher = the right answer appears earlier.',
    example: 'First correct doc at rank 2 → 1/2 = 0.5 for that query; then averaged over all queries.',
  },
  {
    key: 'ndcg',
    name: 'nDCG',
    short: 'Overall ranking quality, top-weighted.',
    long: 'Normalized Discounted Cumulative Gain — rewards putting relevant docs near the top, scaled 0–1 against the ideal ordering.',
    example: 'Correct docs at ranks 1 & 2 score higher than the same docs sitting at ranks 4 & 5.',
  },
  {
    key: 'map',
    name: 'MAP',
    short: 'Ranking quality across all correct sources.',
    long: 'Mean Average Precision — for each query, average the precision measured at every relevant hit, then take the mean over queries. Rewards ranking all relevant docs high, not just the first.',
    example: 'Relevant hits at ranks 1 and 3 → avg of P@1=1.0 and P@3=0.67 = 0.83 for that query.',
  },
  {
    key: 'latency',
    name: 'Latency',
    short: 'Time per query.',
    long: 'Wall-clock retrieval time. p50 = median; p95 = slow tail (95% of queries finish faster than this).',
    example: 'p50 40ms means half the queries finished in under 40ms.',
  },
  {
    key: 'ci',
    name: '95% CI',
    short: 'How trustworthy is the number?',
    long: '95% bootstrap confidence interval — the queries are resampled many times to estimate the range the true metric likely sits in. Narrow = stable; wide = few or noisy cases.',
    example: 'nDCG 0.62 with CI 0.55–0.69: on similar data, expect roughly that band.',
  },
]

export const METRIC_TITLE: Record<string, string> = Object.fromEntries(METRIC_DEFS.map((d) => [d.key, `${d.name} — ${d.short}`]))
