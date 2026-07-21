import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { endpoints, errorMessage } from '../../api/client'
import type { IndexInfo, IndexListEntry, IndicesOverviewResponse } from '../../api/types'
import { Button } from '../../components/Button'
import { BucketQuickPick } from '../../components/BucketQuickPick'
import { Checkbox, Field, Select, TextInput } from '../../components/Field'
import { Dropzone } from '../../components/Dropzone'
import {
  AlertTriangleIcon,
  BookOpenIcon,
  DatabaseIcon,
  FileTextIcon,
  FolderIcon,
  LockIcon,
  PackageIcon,
  PlusIcon,
  TagIcon,
} from '../../components/icons'
import { JobsSection } from '../../components/JobsSection'
import { KbOverview } from '../../components/KbOverview'
import { MetadataRows, metaRowsToObject, type MetaRow } from '../../components/MetadataRows'
import { ProgressBar } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { PageHeader } from '../../components/PageHeader'
import { Pill } from '../../components/Pill'
import { runIngestBatch } from '../../lib/ingestBatch'
import { chunkingSummary } from '../../lib/format'
import { useHistoryStore } from '../../store/historyStore'
import { toast } from '../../store/toastStore'

export function IngestionPage() {
  const [vectorStore, setVectorStore] = useState('')
  const [indices, setIndices] = useState<IndexListEntry[]>([])
  const [overview, setOverview] = useState<IndicesOverviewResponse | null>(null)
  const knownBuckets = useHistoryStore((s) => s.buckets)

  const [indexName, setIndexName] = useState('')
  const [lockedMeta, setLockedMeta] = useState<IndexInfo | null>(null)
  const [bucket, setBucket] = useState('')
  const [metaRows, setMetaRows] = useState<MetaRow[]>([])
  const [enrich, setEnrich] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [progress, setProgress] = useState<{ pct: number; label: string } | null>(null)

  useEffect(() => {
    endpoints.config().then((c) => setVectorStore(c.vector_store || '')).catch(() => {})
    endpoints.indices().then((r) => setIndices(r.indices)).catch(() => {})
    endpoints.indicesOverview().then(setOverview).catch(() => {})
  }, [])

  useEffect(() => {
    if (!indexName) {
      setLockedMeta(null)
      return
    }
    endpoints
      .indexInfo(indexName)
      .then(setLockedMeta)
      .catch((e) => toast(errorMessage(e), 'error'))
  }, [indexName])

  const bucketFiles = lockedMeta?.bucket_files ?? {}
  const bucketNames = Object.keys(bucketFiles)
  const filesInSelectedBucket = bucket && bucketNames.includes(bucket) ? bucketFiles[bucket] || [] : []

  async function submit() {
    if (!files.length) {
      toast('Choose a file first', 'error')
      return
    }
    if (!indexName || !lockedMeta) {
      toast('Pick an index to add to', 'error')
      return
    }
    const chunkParams: Record<string, number> = {}
    if (lockedMeta.chunking) {
      for (const [k, v] of Object.entries(lockedMeta.chunking)) if (k !== 'strategy') chunkParams[k] = Number(v)
    }
    const metadata: Record<string, unknown> = { ...metaRowsToObject(metaRows) }
    if (bucket.trim()) metadata.bucket = bucket.trim()

    setSubmitting(true)
    await runIngestBatch(
      {
        files,
        indexName,
        bucket: bucket.trim(),
        strategy: lockedMeta.chunking?.strategy || null,
        embProvider: lockedMeta.embedding_provider || null,
        embModel: lockedMeta.embedding_model || null,
        chunkParams,
        enrich,
        metadata,
      },
      {
        showProgress: (pct, label) => setProgress({ pct, label }),
        hideProgress: () => setProgress(null),
        onBatchDone: () => setSubmitting(false),
      },
    )
  }

  return (
    <div>
      <PageHeader icon={PackageIcon} title="Ingestion KB">
        Add documents to an existing index. Its chunking and embedding are locked to whatever built it &mdash; to create a new index, use the{' '}
        <Link to="/indices" className="text-signal hover:underline">
          Indices
        </Link>{' '}
        page.
      </PageHeader>
      <div className="flex gap-2 my-4">
        <Pill tone="ok">
          <DatabaseIcon size={11} /> Active vector store: {vectorStore || 'unknown'}
        </Pill>
      </div>

      <Panel
        title={
          <>
            <BookOpenIcon size={14} className="inline -mt-0.5 mr-1" /> Existing knowledge bases{' '}
            <span className="text-ink-faint font-normal text-xs">&mdash; across all vector stores</span>
          </>
        }
        subtitle="What is already indexed. Click a bucket to list its documents."
      >
        <KbOverview overview={overview} />
      </Panel>

      <Panel
        title={
          <>
            <span className="step-badge">1</span> <FolderIcon size={14} className="inline -mt-0.5 mr-1" /> Pick index
          </>
        }
      >
        <Field label="Existing index" hint="its chunking + embedding lock to what built it">
          <Select value={indexName} onChange={(e) => setIndexName(e.target.value)}>
            <option value="">{indices.length ? '(choose an index)' : 'no indices yet — create one on the Indices page'}</option>
            {indices.map((ix) => (
              <option key={ix.index} value={ix.index}>
                {ix.index} ({ix['docs.count']} docs)
              </option>
            ))}
          </Select>
        </Field>
        {lockedMeta && (
          <div className="mt-3 p-3.5 border border-line border-l-2 border-l-signal rounded-panel bg-surface-2/50 text-sm text-ink-dim">
            {lockedMeta.chunking ? (
              <>
                <LockIcon size={12} className="inline -mt-0.5 mr-1" /> Locked to this index's build config:
                <br />
                <strong>chunking</strong> {chunkingSummary(lockedMeta.chunking)}
                <br />
                <strong>embedding</strong> {lockedMeta.embedding_provider || '?'}/{lockedMeta.embedding_model || '?'}
              </>
            ) : (
              <>
                <AlertTriangleIcon size={12} className="inline -mt-0.5 mr-1" /> Legacy index with no recorded build config &mdash; documents
                will be added with the server's current defaults.
              </>
            )}
          </div>
        )}
      </Panel>

      <Panel
        title={
          <>
            <span className="step-badge">2</span> <PackageIcon size={14} className="inline -mt-0.5 mr-1" /> Bucket &amp; files
          </>
        }
      >
        <Field label="Bucket / collection" hint="per-document — you can add a new bucket to an existing index">
          <TextInput placeholder="e.g. triage-protocols, hr-policies" value={bucket} onChange={(e) => setBucket(e.target.value)} />
          <BucketQuickPick buckets={knownBuckets} onPick={setBucket} />
        </Field>
        {lockedMeta && (
          <div>
            {bucketNames.length ? (
              <label className="block text-xs text-ink-faint mb-1">
                Or pick an existing bucket <span className="text-ink-faint">&mdash; shows its docs; already-listed files skip as duplicates</span>
                <Select className="mt-1" value={bucket} onChange={(e) => setBucket(e.target.value)}>
                  <option value="">(choose a bucket)</option>
                  {bucketNames.map((b) => (
                    <option key={b} value={b}>
                      {b} ({bucketFiles[b]?.length ?? 0})
                    </option>
                  ))}
                </Select>
                <ul className="list-none m-0 mt-1.5 p-0 text-xs text-ink-dim flex flex-col gap-0.5">
                  {filesInSelectedBucket.map((f) => (
                    <li key={f} className="flex items-center gap-1.5">
                      <FileTextIcon size={12} className="text-ink-faint" /> {f}
                    </li>
                  ))}
                </ul>
              </label>
            ) : (
              <div className="text-ink-faint text-sm mt-1.5">No documents yet &mdash; this index is empty.</div>
            )}
          </div>
        )}
      </Panel>

      <Panel
        title={
          <>
            <span className="step-badge">3</span> <TagIcon size={14} className="inline -mt-0.5 mr-1" /> Extra metadata &amp; enrichment
          </>
        }
      >
        <MetadataRows rows={metaRows} onChange={setMetaRows} />
        <div className="mt-3">
          <Checkbox
            label={<>LLM keyword enrichment <span className="text-ink-faint text-xs">(slower — adds search keywords per chunk)</span></>}
            checked={enrich}
            onChange={(e) => setEnrich(e.target.checked)}
          />
        </div>
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

      {lockedMeta && (
        <div className="mb-4 p-4 border border-line border-l-2 border-l-accent rounded-panel bg-surface-2/50">
          <div className="font-bold text-sm mb-2 flex items-center gap-1.5">
            <PlusIcon size={13} className="text-accent" /> Adding to <code className="bg-surface-2 px-1.5 py-0.5 rounded">{lockedMeta.index}</code>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink-dim">
            <span>
              <strong>vector store</strong> {vectorStore || '?'}
            </span>
            <span>
              <strong>chunking</strong> {chunkingSummary(lockedMeta.chunking)}
            </span>
            <span>
              <strong>embedding</strong> {lockedMeta.embedding_provider || '?'}/{lockedMeta.embedding_model || '?'}
            </span>
            <span>
              <strong>dim</strong> {lockedMeta.dim ?? '?'}
            </span>
            <span>
              <strong>docs</strong> {lockedMeta.docs_count ?? '?'}
            </span>
            <span>
              <strong>buckets</strong> {lockedMeta.buckets.join(', ') || '—'}
            </span>
          </div>
        </div>
      )}

      <div className="flex items-center gap-4 mb-2">
        <Button variant="primary" onClick={submit} disabled={submitting}>
          Upload &amp; ingest
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
