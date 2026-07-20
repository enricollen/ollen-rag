import { useEffect, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { ConfigResponse, IndexInfo, IndexListEntry } from '../../api/types'
import { Button } from '../../components/Button'
import { Field, TextInput } from '../../components/Field'
import { filterRowsToSpecs, FilterBuilder, type FilterRow } from '../../components/FilterBuilder'
import { IndexInfoPanel } from '../../components/IndexInfoPanel'
import { IndexSelect } from '../../components/IndexSelect'
import { MessageSquareIcon } from '../../components/icons'
import { EmptyState, Spinner } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { PageHeader } from '../../components/PageHeader'
import { RerankerSelect, type RerankerSelection } from '../../components/RerankerSelect'
import { useHistoryStore } from '../../store/historyStore'
import { toast } from '../../store/toastStore'
import { QaItem } from './QaItem'

export function QueryPage() {
  const [indices, setIndices] = useState<IndexListEntry[]>([])
  const [cfg, setCfg] = useState<ConfigResponse | null>(null)
  const [indexName, setIndexName] = useState('')
  const [indexInfo, setIndexInfo] = useState<IndexInfo | null>(null)
  const [selectedBucket, setSelectedBucket] = useState('')
  const [query, setQuery] = useState('')
  const [promptName, setPromptName] = useState('')
  const [filters, setFilters] = useState<FilterRow[]>([])
  const [condition, setCondition] = useState<'and' | 'or'>('and')
  const [reranker, setReranker] = useState<RerankerSelection>({ reranker_provider: null, reranker_model: null })
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const qa = useHistoryStore((s) => s.qa)
  const addQa = useHistoryStore((s) => s.addQa)
  const clearQa = useHistoryStore((s) => s.clearQa)

  useEffect(() => {
    endpoints
      .indices()
      .then((ixs) => {
        setIndices(ixs.indices)
        if (ixs.indices.length) setIndexName((prev) => prev || ixs.indices[0].index)
      })
      .catch(() => {})
    endpoints.config().then(setCfg).catch(() => {})
  }, [])

  useEffect(() => {
    if (!indexName) {
      setIndexInfo(null)
      return
    }
    setSelectedBucket('')
    endpoints
      .indexInfo(indexName)
      .then(setIndexInfo)
      .catch(() => setIndexInfo(null))
  }, [indexName])

  async function submit() {
    if (!query.trim()) {
      toast('Enter a question', 'error')
      return
    }
    if (!indexName) {
      toast('No index selected', 'error')
      return
    }
    const specs = filterRowsToSpecs(filters)
    if (selectedBucket) specs.unshift({ key: 'bucket', value: selectedBucket, operator: '==' })

    setStatus('generating…')
    setLoading(true)
    try {
      const res = await endpoints.query({
        query,
        strategy: null,
        index_name: indexName,
        prompt_name: promptName.trim() || null,
        filters: specs.length ? specs : null,
        filter_condition: condition,
        ...reranker,
      })
      addQa({ query, answer: res.answer, sources: res.sources })
      setStatus('')
      setQuery('')
    } catch (e) {
      toast(errorMessage(e), 'error')
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <PageHeader icon={MessageSquareIcon} title="Query (end-to-end)">
        Retrieval + rerank + cited LLM answer via CitationQueryEngine. Inline{' '}
        <code className="bg-surface-2 px-1.5 py-0.5 rounded">[n]</code> citations link to the exact source chunk below.
      </PageHeader>

      <Panel>
        <Field label="Question">
          <TextInput placeholder="what are the triage color codes?" value={query} onChange={(e) => setQuery(e.target.value)} />
        </Field>
        <div className="flex gap-4">
          <Field label="Index" className="flex-1">
            <IndexSelect indices={indices} value={indexName} onChange={setIndexName} />
          </Field>
          <Field label="Prompt name (optional)" className="flex-1">
            <TextInput placeholder="rag_answer" value={promptName} onChange={(e) => setPromptName(e.target.value)} />
          </Field>
        </div>
        {indexInfo && <IndexInfoPanel info={indexInfo} verb="Queries" onBucketSelect={setSelectedBucket} />}
        <Field label="Additional metadata filters (optional)">
          <FilterBuilder rows={filters} onChange={setFilters} condition={condition} onConditionChange={setCondition} />
        </Field>
        <details open={advancedOpen} onToggle={(e) => setAdvancedOpen((e.target as HTMLDetailsElement).open)} className="mt-3">
          <summary className="text-ink-faint text-sm cursor-pointer">Advanced: reranker</summary>
          {cfg && (
            <Field label="Reranker" className="mt-2">
              <RerankerSelect cfg={cfg} value={reranker} onChange={setReranker} />
            </Field>
          )}
        </details>
        <div className="flex items-center gap-4 mt-4">
          <Button variant="primary" onClick={submit} disabled={loading}>
            {loading && <Spinner />} Ask
          </Button>
          <span className="text-xs text-ink-faint">{status}</span>
        </div>
      </Panel>

      <div className="flex justify-end mt-4 mb-4">
        <Button
          variant="secondary"
          onClick={() => {
            clearQa()
            toast('Query history cleared', 'info')
          }}
        >
          Clear history
        </Button>
      </div>

      {qa.length ? (
        qa.map((entry, i) => <QaItem key={i} entry={entry} />)
      ) : (
        <EmptyState>Ask a question to see a cited answer here.</EmptyState>
      )}
    </div>
  )
}
