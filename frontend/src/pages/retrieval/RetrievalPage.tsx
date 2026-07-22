import { useEffect, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { ConfigResponse, IndexInfo, IndexListEntry, RetrieveResponse } from '../../api/types'
import { Button } from '../../components/Button'
import { Field, TextInput } from '../../components/Field'
import { filterRowsToSpecs, FilterBuilder, type FilterRow } from '../../components/FilterBuilder'
import { IndexInfoPanel } from '../../components/IndexInfoPanel'
import { IndexSelect, defaultIndexName } from '../../components/IndexSelect'
import { SearchIcon } from '../../components/icons'
import { Spinner } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { PageHeader } from '../../components/PageHeader'
import { Pill } from '../../components/Pill'
import { RerankerSelect, type RerankerSelection } from '../../components/RerankerSelect'
import { useHistoryStore } from '../../store/historyStore'
import { toast } from '../../store/toastStore'
import { ResultColumn } from './ResultColumn'

export function RetrievalPage() {
  const [indices, setIndices] = useState<IndexListEntry[]>([])
  const [cfg, setCfg] = useState<ConfigResponse | null>(null)
  const [indexName, setIndexName] = useState('')
  const [indexInfo, setIndexInfo] = useState<IndexInfo | null>(null)
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState('')
  const [rerankTopN, setRerankTopN] = useState('')
  const [threshold, setThreshold] = useState('')
  const [filters, setFilters] = useState<FilterRow[]>([])
  const [condition, setCondition] = useState<'and' | 'or'>('and')
  const [reranker, setReranker] = useState<RerankerSelection>({ reranker_provider: null, reranker_model: null })
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<RetrieveResponse | null>(null)
  const buckets = useHistoryStore((s) => s.buckets)

  useEffect(() => {
    Promise.all([endpoints.config(), endpoints.indices()])
      .then(([c, ixs]) => {
        setCfg(c)
        setIndices(ixs.indices)
        setTopK(String(c.retrieval_top_k ?? ''))
        setRerankTopN(String(c.rerank_top_n ?? ''))
        setIndexName(defaultIndexName(ixs.indices))
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!indexName) {
      setIndexInfo(null)
      return
    }
    endpoints
      .indexInfo(indexName)
      .then(setIndexInfo)
      .catch(() => setIndexInfo(null))
  }, [indexName])

  async function submit() {
    if (!query.trim()) {
      toast('Enter a query', 'error')
      return
    }
    if (!indexName) {
      toast('No index selected', 'error')
      return
    }
    setStatus('')
    setLoading(true)
    setResults(null)
    try {
      const res = await endpoints.retrieve({
        query,
        strategy: null,
        index_name: indexName,
        top_k: topK ? Number(topK) : null,
        rerank_top_n: rerankTopN ? Number(rerankTopN) : null,
        similarity_threshold: threshold !== '' ? Number(threshold) : null,
        filters: filterRowsToSpecs(filters).length ? filterRowsToSpecs(filters) : null,
        filter_condition: condition,
        ...reranker,
      })
      const hybridCount = res.hybrid_nodes?.length ?? 0
      setStatus(`${hybridCount} post-threshold → ${res.nodes.length} after rerank`)
      setResults(res)
    } catch (e) {
      toast(errorMessage(e), 'error')
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <PageHeader icon={SearchIcon} title="Retrieval">
        Hybrid BM25 + dense retrieval (native OpenSearch), reranked by a cross-encoder. Pick an index, inspect its documents, and test filters
        without invoking the LLM.
      </PageHeader>

      <Panel>
        <Field label="Query">
          <TextInput placeholder="what is triage?" value={query} onChange={(e) => setQuery(e.target.value)} />
        </Field>
        <Field label="Index">
          <IndexSelect indices={indices} value={indexName} onChange={setIndexName} />
        </Field>
        {indexInfo && <IndexInfoPanel info={indexInfo} verb="Retrievals" />}
        <div className="flex gap-4">
          <Field label="top_k" className="flex-1">
            <TextInput type="number" min={1} max={100} value={topK} onChange={(e) => setTopK(e.target.value)} />
          </Field>
          <Field label="rerank_top_n" className="flex-1">
            <TextInput type="number" min={1} max={50} value={rerankTopN} onChange={(e) => setRerankTopN(e.target.value)} />
          </Field>
          <Field label="similarity_threshold (optional override)" className="flex-1">
            <TextInput type="number" min={0} max={1} step="0.01" placeholder={String(cfg?.similarity_threshold ?? '')} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
          </Field>
          {cfg && (
            <Field label="Reranker" className="flex-1">
              <RerankerSelect cfg={cfg} value={reranker} onChange={setReranker} />
            </Field>
          )}
        </div>

        <Field label="Metadata filters">
          {buckets.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-2">
              {buckets.map((b) => (
                <Pill key={b} onClick={() => setFilters([...filters, { key: 'bucket', value: b, operator: '==' }])}>
                  bucket = {b}
                </Pill>
              ))}
            </div>
          )}
          <FilterBuilder rows={filters} onChange={setFilters} condition={condition} onConditionChange={setCondition} />
        </Field>

        <div className="flex items-center gap-4 mt-4">
          <Button variant="primary" onClick={submit} disabled={loading}>
            {loading && <Spinner />} Retrieve
          </Button>
          <span className="text-xs text-ink-faint">{status}</span>
        </div>
      </Panel>

      {results && (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-4 mt-4">
          <ResultColumn title="BM25" subtitle="Lexical-only hits (OpenSearch text search)." nodes={results.bm25_nodes || []} />
          <ResultColumn title="Dense" subtitle="Embedding-only hits (kNN vector search)." nodes={results.dense_nodes || []} />
          <ResultColumn
            title="Hybrid (post-threshold)"
            subtitle="Fused BM25+dense scores after similarity_threshold filter, before reranking."
            nodes={results.hybrid_nodes || []}
          />
          <ResultColumn
            title="Cross-encoder (final)"
            subtitle="Hybrid BM25+dense, fused then reranked — what /query uses."
            nodes={results.nodes}
          />
        </div>
      )}
    </div>
  )
}
