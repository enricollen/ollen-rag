import { useEffect, useMemo, useState } from 'react'
import { endpoints } from '../../api/client'
import type { ConfigResponse, IndexListEntry } from '../../api/types'
import { Button } from '../../components/Button'
import { BucketQuickPick } from '../../components/BucketQuickPick'
import { Checkbox, Field, Select, TextInput } from '../../components/Field'
import { Dropzone } from '../../components/Dropzone'
import { AlertTriangleIcon, CpuIcon, DatabaseIcon, FileTextIcon, PackageIcon, ScissorsIcon, SparklesIcon, TagIcon } from '../../components/icons'
import { JobsSection } from '../../components/JobsSection'
import { MetadataRows, metaRowsToObject, type MetaRow } from '../../components/MetadataRows'
import { ProgressBar } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { Pill } from '../../components/Pill'
import { ChunkParamInputs, StrategyPicker } from '../../components/StrategyPicker'
import { runIngestBatch } from '../../lib/ingestBatch'
import { CHUNK_FIELDS } from '../../lib/strategies'
import { useHistoryStore } from '../../store/historyStore'
import { toast } from '../../store/toastStore'

export function CreateIndexTab() {
  const [cfg, setCfg] = useState<ConfigResponse | null>(null)
  const [indices, setIndices] = useState<IndexListEntry[]>([])
  const knownBuckets = useHistoryStore((s) => s.buckets)

  const [indexName, setIndexName] = useState('')
  const [indexNameEdited, setIndexNameEdited] = useState(false)
  const [bucket, setBucket] = useState('')
  const [strategy, setStrategy] = useState('sentence')
  const [chunkParams, setChunkParams] = useState<Record<string, number | undefined>>({})
  const [enrich, setEnrich] = useState(false)
  const [embProvider, setEmbProvider] = useState('')
  const [embModel, setEmbModel] = useState('')
  const [metaRows, setMetaRows] = useState<MetaRow[]>([])
  const [files, setFiles] = useState<FileList | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [progress, setProgress] = useState<{ pct: number; label: string } | null>(null)

  useEffect(() => {
    endpoints
      .config()
      .then((c) => {
        setCfg(c)
        setStrategy(c.default_chunking_strategy)
        setEmbProvider(c.embedding_provider || Object.keys(c.embedding_model_choices)[0] || '')
        setChunkParams({
          chunk_size: c.chunk_size,
          chunk_overlap: c.chunk_overlap,
          semantic_breakpoint_percentile: c.semantic_breakpoint_percentile,
          sentence_window_size: c.sentence_window_size,
          llm_chunk_max_size: c.llm_chunk_max_size,
          llm_chunk_window_size: c.llm_chunk_window_size,
        })
      })
      .catch(() => {})
    endpoints
      .indices()
      .then((r) => setIndices(r.indices))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!indexNameEdited) setIndexName(strategy)
  }, [strategy, indexNameEdited])

  const embeddingChoices = cfg?.embedding_model_choices ?? {}
  const embeddingDefaults = cfg?.embedding_default_models ?? {}
  const modelsForProvider = embeddingChoices[embProvider] ?? []

  useEffect(() => {
    const def = embeddingDefaults[embProvider] || ''
    if (modelsForProvider.length) {
      setEmbModel(modelsForProvider.includes(def) ? def : modelsForProvider[0])
    } else {
      setEmbModel(def)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [embProvider, cfg])

  const existingIndexNames = useMemo(() => new Set(indices.map((ix) => ix.index)), [indices])
  const finalName = indexName.trim() || strategy || '(unnamed)'
  const nameExists = existingIndexNames.has(finalName)

  async function submit() {
    if (!files || !files.length) {
      toast('Choose a file first', 'error')
      return
    }
    if (!finalName) {
      toast('Enter an index name', 'error')
      return
    }
    if (nameExists) {
      toast(`Index "${finalName}" already exists — add to it from the Ingestion KB page`, 'error')
      return
    }
    if (!embModel.trim()) {
      toast('Enter an embedding model', 'error')
      return
    }
    const metadata: Record<string, unknown> = { ...metaRowsToObject(metaRows) }
    if (bucket.trim()) metadata.bucket = bucket.trim()

    const numericChunkParams: Record<string, number> = {}
    for (const f of CHUNK_FIELDS[strategy] || []) {
      const v = chunkParams[f.key]
      if (v !== undefined) numericChunkParams[f.key] = v
    }

    setSubmitting(true)
    await runIngestBatch(
      {
        files,
        indexName: finalName,
        bucket: bucket.trim(),
        strategy,
        embProvider,
        embModel: embModel.trim(),
        chunkParams: numericChunkParams,
        enrich,
        metadata,
      },
      {
        showProgress: (pct, label) => setProgress({ pct, label }),
        hideProgress: () => setProgress(null),
        onBatchDone: () => {
          setSubmitting(false)
          endpoints.indices().then((r) => setIndices(r.indices))
        },
      },
    )
  }

  return (
    <div>
      <div className="flex gap-2 mb-5">
        <Pill tone="ok">
          <DatabaseIcon size={11} /> Active vector store: {cfg?.vector_store || 'unknown'}
        </Pill>
      </div>

      <Panel
        title={
          <>
            <span className="step-badge">1</span> <PackageIcon size={14} className="inline -mt-0.5 mr-1" /> Index &amp; bucket
          </>
        }
      >
        <Field label="Index name" hint="new index; suggested from strategy, editable">
          <TextInput
            placeholder="sentence"
            value={indexName}
            onChange={(e) => {
              setIndexName(e.target.value)
              setIndexNameEdited(true)
            }}
          />
        </Field>
        <Field label="Bucket / collection" hint="stored as metadata.bucket — filter on it later in Retrieval/Query">
          <TextInput placeholder="e.g. triage-protocols, hr-policies" value={bucket} onChange={(e) => setBucket(e.target.value)} />
          <BucketQuickPick buckets={knownBuckets} onPick={setBucket} />
        </Field>
      </Panel>

      <Panel
        title={
          <>
            <span className="step-badge">2</span> <ScissorsIcon size={14} className="inline -mt-0.5 mr-1" /> Chunking
          </>
        }
      >
        <StrategyPicker selected={strategy} onSelect={setStrategy} />
        <ChunkParamInputs strategy={strategy} values={chunkParams} onChange={setChunkParams} />
        <div className="mt-3">
          <Checkbox
            label={<>LLM keyword enrichment <span className="text-ink-faint text-xs">(slower — one LLM call per chunk; adds search keywords to boost recall)</span></>}
            checked={enrich}
            onChange={(e) => setEnrich(e.target.checked)}
          />
        </div>
      </Panel>

      <Panel
        title={
          <>
            <span className="step-badge">3</span> <CpuIcon size={14} className="inline -mt-0.5 mr-1" /> Embedding model
          </>
        }
      >
        <Field label="Provider">
          <Select value={embProvider} onChange={(e) => setEmbProvider(e.target.value)}>
            {Object.keys(embeddingChoices).map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Model">
          {modelsForProvider.length ? (
            <Select value={embModel} onChange={(e) => setEmbModel(e.target.value)}>
              {modelsForProvider.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </Select>
          ) : (
            <TextInput
              placeholder="any LiteLLM embedding model, e.g. cohere/embed-multilingual-v3.0"
              value={embModel}
              onChange={(e) => setEmbModel(e.target.value)}
            />
          )}
        </Field>
      </Panel>

      <Panel
        title={
          <>
            <span className="step-badge">4</span> <TagIcon size={14} className="inline -mt-0.5 mr-1" /> Extra metadata{' '}
            <span className="text-ink-faint text-xs font-normal">(optional, merged into every chunk alongside bucket)</span>
          </>
        }
      >
        <MetadataRows rows={metaRows} onChange={setMetaRows} />
      </Panel>

      <Panel
        title={
          <>
            <FileTextIcon size={14} className="inline -mt-0.5 mr-1" /> Document(s)
          </>
        }
      >
        <Dropzone onFiles={setFiles} />
      </Panel>

      <div className="mb-4 p-4 border border-line border-l-2 border-l-accent rounded-panel bg-surface-2/50">
        <div className="font-bold text-sm mb-2 flex items-center gap-1.5">
          <SparklesIcon size={14} className="text-accent" /> New index
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink-dim">
          <span>
            <strong>index</strong> <code className="bg-surface-2 px-1.5 py-0.5 rounded">{finalName}</code>
          </span>
          <span>
            <strong>vector store</strong> {cfg?.vector_store || '?'}
          </span>
          <span>
            <strong>chunking</strong> {[strategy, ...(CHUNK_FIELDS[strategy] || []).map((f) => `${f.key}=${chunkParams[f.key] ?? ''}`)].join(' · ')}
          </span>
          <span>
            <strong>embedding</strong> {embProvider}/{embModel}
          </span>
        </div>
        {nameExists && (
          <div className="text-warn text-sm mt-2 flex items-center gap-1.5">
            <AlertTriangleIcon size={13} /> An index named "{finalName}" already exists &mdash; add to it from the Ingestion KB page, or
            choose a different name.
          </div>
        )}
      </div>

      <div className="flex items-center gap-4 mb-2">
        <Button variant="primary" onClick={submit} disabled={submitting || nameExists}>
          Create index &amp; ingest
        </Button>
      </div>
      {progress && (
        <div className="mb-6">
          <ProgressBar pct={progress.pct} />
          <div className="text-xs text-ink-dim mt-1">{progress.label}</div>
        </div>
      )}

      <JobsSection />
    </div>
  )
}
