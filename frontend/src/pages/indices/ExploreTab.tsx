import { useEffect, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { IndexDocument, IndicesOverviewResponse } from '../../api/types'
import { Button } from '../../components/Button'
import { ChunkText } from '../../components/ChunkText'
import { KbOverview } from '../../components/KbOverview'
import { BookOpenIcon, ChevronLeftIcon, ChevronRightIcon, FileTextIcon, FolderIcon, PackageIcon, TrashIcon } from '../../components/icons'
import { EmptyState, Spinner } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { Pill } from '../../components/Pill'
import { toast } from '../../store/toastStore'

const PAGE_SIZE = 20

function DocCard({ doc }: { doc: IndexDocument }) {
  const chips = Object.entries(doc.metadata || {}).filter(([k]) => !['_node_type', '_node_content'].includes(k))
  return (
    <div className="bg-surface-2/60 border border-line rounded-panel p-3.5 mb-3">
      <div className="text-ink-faint text-xs mb-1.5">
        id: <code className="bg-surface-2 px-1.5 py-0.5 rounded font-mono">{doc.id}</code>
      </div>
      <ChunkText text={doc.content} />
      <div className="flex flex-wrap gap-1.5 mt-2">
        {chips.map(([k, v]) => (
          <Pill key={k}>
            {k}: {String(v).slice(0, 60)}
          </Pill>
        ))}
      </div>
    </div>
  )
}

export function ExploreTab() {
  const [overview, setOverview] = useState<IndicesOverviewResponse | null>(null)
  const [currentIndex, setCurrentIndex] = useState<string | null>(null)
  const [bucketFiles, setBucketFiles] = useState<Record<string, string[]>>({})
  const [unbucketedFiles, setUnbucketedFiles] = useState<string[]>([])
  const [currentBucket, setCurrentBucket] = useState<string | null>(null)
  const [currentUnbucketed, setCurrentUnbucketed] = useState(false)
  const [currentFileName, setCurrentFileName] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [docs, setDocs] = useState<IndexDocument[]>([])
  const [total, setTotal] = useState(0)
  const [loadingDocs, setLoadingDocs] = useState(false)
  const [loadingBuckets, setLoadingBuckets] = useState(false)

  async function refreshOverview() {
    try {
      const ov = await endpoints.indicesOverview()
      setOverview(ov)
      const stillExists = ov.stores.some((s) => s.active && s.indices.some((i) => i.index === currentIndex))
      if (!stillExists) setCurrentIndex(null)
    } catch (e) {
      toast(errorMessage(e), 'error')
    }
  }

  useEffect(() => {
    refreshOverview()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!currentIndex) {
      setBucketFiles({})
      setUnbucketedFiles([])
      return
    }
    setLoadingBuckets(true)
    endpoints
      .indexInfo(currentIndex)
      .then((info) => {
        setBucketFiles(info.bucket_files || {})
        setUnbucketedFiles(info.unbucketed_files || [])
      })
      .catch((e) => toast(errorMessage(e), 'error'))
      .finally(() => setLoadingBuckets(false))
  }, [currentIndex])

  useEffect(() => {
    if (!currentIndex) {
      setDocs([])
      return
    }
    setLoadingDocs(true)
    endpoints
      .indexDocuments(currentIndex, {
        offset: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        bucket: currentBucket,
        unbucketed: currentUnbucketed,
        file_name: currentFileName,
      })
      .then((res) => {
        setDocs(res.documents)
        setTotal(res.total)
      })
      .catch((e) => toast(errorMessage(e), 'error'))
      .finally(() => setLoadingDocs(false))
  }, [currentIndex, page, currentBucket, currentUnbucketed, currentFileName])

  function selectIndex(_store: string, index: string) {
    setCurrentIndex(index)
    setPage(0)
    setCurrentBucket(null)
    setCurrentUnbucketed(false)
    setCurrentFileName(null)
  }

  async function deleteIndex(_store: string, index: string) {
    const typed = prompt(`This permanently deletes ALL documents in "${index}". This cannot be undone.\n\nType the index name to confirm:`)
    if (typed !== index) {
      if (typed !== null) toast('Index name did not match — nothing deleted', 'error')
      return
    }
    try {
      await endpoints.deleteIndex(index)
      toast(`Deleted index "${index}"`, 'success')
      if (currentIndex === index) setCurrentIndex(null)
      await refreshOverview()
    } catch (e) {
      toast(errorMessage(e), 'error')
    }
  }

  async function deleteBucket(bucket: string, count: number) {
    if (!currentIndex) return
    if (!confirm(`Delete bucket "${bucket}" and its ${count} document(s) from "${currentIndex}"? This cannot be undone.`)) return
    try {
      const res = await endpoints.deleteBucket(currentIndex, bucket)
      toast(`Deleted bucket "${bucket}" (${res.deleted} document(s))`, 'success')
      if (currentBucket === bucket) {
        setCurrentBucket(null)
        setCurrentFileName(null)
      }
      setPage(0)
      const info = await endpoints.indexInfo(currentIndex)
      setBucketFiles(info.bucket_files || {})
      setUnbucketedFiles(info.unbucketed_files || [])
      refreshOverview()
    } catch (e) {
      toast(errorMessage(e), 'error')
    }
  }

  function toggleBucket(bucket: string | null, unbucketed: boolean) {
    const already = unbucketed ? currentUnbucketed : currentBucket === bucket
    setCurrentBucket(already || unbucketed ? null : bucket)
    setCurrentUnbucketed(!already && unbucketed)
    setCurrentFileName(null)
    setPage(0)
  }

  const bucketNames = Object.keys(bucketFiles)
  const activeFiles = currentUnbucketed ? unbucketedFiles : currentBucket ? bucketFiles[currentBucket] || [] : null
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const scopeLabel =
    (currentUnbucketed ? ' with no bucket' : currentBucket ? ` in bucket "${currentBucket}"` : ' in this index') +
    (currentFileName ? ` from "${currentFileName}"` : '')

  return (
    <div>
      <Panel
        title={
          <>
            <BookOpenIcon size={14} className="text-signal inline -mt-0.5 mr-1" /> Existing indexes{' '}
            <span className="text-ink-faint font-normal text-xs">&mdash; across all vector stores</span>
          </>
        }
        badge={
          <Button variant="secondary" onClick={refreshOverview} className="ml-auto">
            Refresh
          </Button>
        }
      >
        <KbOverview overview={overview} selectable deletable selectedIndex={currentIndex} onSelect={selectIndex} onDelete={deleteIndex} />
      </Panel>

      <Panel title="Buckets" subtitle="Select a bucket to see the documents it contains. Documents ingested without a bucket appear under No bucket.">
        {!currentIndex ? (
          <EmptyState>Select an index above to browse its buckets.</EmptyState>
        ) : loadingBuckets ? (
          <EmptyState>
            <Spinner />
          </EmptyState>
        ) : !bucketNames.length && !unbucketedFiles.length ? (
          <EmptyState>No documents in this index.</EmptyState>
        ) : (
          <div className="flex flex-wrap gap-2.5">
            {bucketNames.map((b) => (
              <div key={b} className="flex flex-col gap-1.5">
                <button
                  type="button"
                  onClick={() => toggleBucket(b, false)}
                  className={`flex flex-col items-start gap-0.5 min-w-[140px] px-3.5 py-2.5 bg-surface-2/60 border rounded-panel transition-colors ${
                    currentBucket === b ? 'border-signal shadow-[inset_0_0_0_1px_var(--color-signal)]' : 'border-line hover:border-accent/60'
                  }`}
                >
                  <PackageIcon size={20} className="text-ink-dim" />
                  <span className="font-semibold text-sm text-ink break-all">{b}</span>
                  <span className="text-xs text-ink-dim">{bucketFiles[b]?.length ?? 0} docs</span>
                </button>
                <button
                  type="button"
                  onClick={() => deleteBucket(b, bucketFiles[b]?.length ?? 0)}
                  className="text-xs text-bad border border-bad rounded px-2 py-1 bg-bad/10 hover:bg-bad hover:text-white inline-flex items-center justify-center gap-1"
                >
                  <TrashIcon size={11} /> Delete
                </button>
              </div>
            ))}
            {unbucketedFiles.length > 0 && (
              <button
                type="button"
                onClick={() => toggleBucket(null, true)}
                className={`flex flex-col items-start gap-0.5 min-w-[140px] px-3.5 py-2.5 bg-surface-2/60 border rounded-panel transition-colors ${
                  currentUnbucketed ? 'border-signal shadow-[inset_0_0_0_1px_var(--color-signal)]' : 'border-line hover:border-accent/60'
                }`}
              >
                <FolderIcon size={20} className="text-ink-dim" />
                <span className="font-semibold text-sm text-ink">No bucket</span>
                <span className="text-xs text-ink-dim">{unbucketedFiles.length} docs</span>
              </button>
            )}
          </div>
        )}
        {activeFiles && (
          <div className="mt-4">
            <div className="text-sm font-semibold text-ink mb-2 flex items-center gap-1.5">
              {currentUnbucketed ? (
                <>
                  <FolderIcon size={13} /> No bucket
                </>
              ) : (
                <>
                  <PackageIcon size={13} /> {currentBucket}
                </>
              )}{' '}
              &mdash; {activeFiles.length} document(s){' '}
              <span className="text-ink-faint text-xs font-normal">(click a file to filter stored chunks below)</span>
            </div>
            <ul className="list-none m-0 p-0 flex flex-col gap-1">
              {activeFiles.length ? (
                activeFiles.map((f) => (
                  <li
                    key={f}
                    onClick={() => {
                      setCurrentFileName(currentFileName === f ? null : f)
                      setPage(0)
                    }}
                    className={`text-sm px-3 py-1.5 rounded-control cursor-pointer transition-colors ${
                      currentFileName === f ? 'border border-signal text-ink shadow-[inset_0_0_0_1px_var(--color-signal)]' : 'border border-line text-ink-dim hover:border-accent/50'
                    }`}
                  >
                    <FileTextIcon size={12} className="inline -mt-0.5 mr-1" /> {f}
                  </li>
                ))
              ) : (
                <li className="text-ink-faint text-sm">empty</li>
              )}
            </ul>
          </div>
        )}
      </Panel>

      <Panel title="All stored chunks">
        <div className="flex justify-between items-center mb-3">
          <span className="text-xs text-ink-faint">
            {currentIndex ? `page ${page + 1} / ${totalPages} · ${total} document(s) total${scopeLabel}` : ''}
          </span>
          <div className="flex gap-2">
            <Button variant="secondary" disabled={!currentIndex || page === 0} onClick={() => setPage((p) => p - 1)}>
              <ChevronLeftIcon size={13} /> Prev
            </Button>
            <Button variant="secondary" disabled={!currentIndex || page + 1 >= totalPages} onClick={() => setPage((p) => p + 1)}>
              Next <ChevronRightIcon size={13} />
            </Button>
          </div>
        </div>
        {!currentIndex ? (
          <EmptyState>Select an index above to browse its stored chunks.</EmptyState>
        ) : loadingDocs ? (
          <EmptyState>
            <Spinner />
          </EmptyState>
        ) : !docs.length ? (
          <EmptyState>No documents{scopeLabel}.</EmptyState>
        ) : (
          docs.map((d) => <DocCard key={d.id} doc={d} />)
        )}
      </Panel>
    </div>
  )
}
