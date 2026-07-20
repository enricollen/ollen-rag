import { useState } from 'react'
import type { IndexOverviewEntry, IndicesOverviewResponse } from '../api/types'
import { chunkingSummary } from '../lib/format'
import { Pill } from './Pill'
import { AlertTriangleIcon, CpuIcon, DatabaseIcon, FileTextIcon, PackageIcon, ScissorsIcon, TrashIcon } from './icons'

function OverviewCard({
  store,
  ix,
  active,
  selectable,
  deletable,
  selected,
  onSelect,
  onDelete,
}: {
  store: string
  ix: IndexOverviewEntry
  active: boolean
  selectable: boolean
  deletable: boolean
  selected: boolean
  onSelect?: (store: string, index: string) => void
  onDelete?: (store: string, index: string) => void
}) {
  const [openBucket, setOpenBucket] = useState<string | null>(null)
  const emb = ix.embedding_provider ? `${ix.embedding_provider}/${ix.embedding_model}` : 'unrecorded'
  const bucketNames = Object.keys(ix.bucket_files || {})
  const manageable = active
  const clickable = selectable && manageable

  return (
    <div
      onClick={clickable ? () => onSelect?.(store, ix.index) : undefined}
      className={`bg-surface-2/60 border rounded-panel p-3.5 transition-colors ${clickable ? 'cursor-pointer' : ''} ${
        selected ? 'border-accent shadow-[inset_0_0_0_1px_var(--color-accent)]' : 'border-line hover:border-accent/50'
      } ${!manageable ? 'opacity-60' : ''}`}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <code className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap font-mono text-[0.82rem] bg-surface px-1.5 py-0.5 rounded">
          {ix.index}
        </code>
        <span className="flex items-center gap-1.5 flex-none">
          <span className="text-[0.72rem] text-ink-faint whitespace-nowrap">{ix.docs_count} docs</span>
          {deletable && manageable && (
            <button
              type="button"
              title="Delete this index"
              onClick={(e) => {
                e.stopPropagation()
                onDelete?.(store, ix.index)
              }}
              className="text-[0.78rem] text-bad border border-bad rounded px-1.5 py-0.5 bg-bad/10 hover:bg-bad hover:text-white whitespace-nowrap inline-flex items-center gap-1"
            >
              <TrashIcon size={11} /> Delete
            </button>
          )}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5 mb-2">
        <Pill tone="soft">
          <CpuIcon size={11} /> {emb}
        </Pill>
        <Pill tone="soft">
          <ScissorsIcon size={11} /> {chunkingSummary(ix.chunking)}
        </Pill>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {bucketNames.length ? (
          bucketNames.map((b) => (
            <Pill
              key={b}
              tone={openBucket === b ? 'ok' : 'default'}
              onClick={(e) => {
                e?.stopPropagation()
                setOpenBucket(openBucket === b ? null : b)
              }}
            >
              <PackageIcon size={11} /> {b} <span className="bg-line rounded-full px-1.5 text-ink-dim">{ix.bucket_files[b]?.length ?? 0}</span>
            </Pill>
          ))
        ) : (
          <span className="text-ink-faint text-xs">no buckets yet</span>
        )}
      </div>
      {openBucket && (
        <ul className="mt-2 pt-2 border-t border-dashed border-line list-none m-0 p-0 flex flex-col gap-0.5 text-xs text-ink-dim">
          {(ix.bucket_files[openBucket] || []).length ? (
            ix.bucket_files[openBucket].map((f) => (
              <li key={f} className="truncate flex items-center gap-1.5">
                <FileTextIcon size={12} className="text-ink-faint" /> {f}
              </li>
            ))
          ) : (
            <li className="text-ink-faint">empty bucket &mdash; no documents</li>
          )}
        </ul>
      )}
      {!manageable && (
        <div className="mt-2 pt-2 border-t border-dashed border-line text-[0.72rem] text-ink-faint">
          inactive store &mdash; switch vector store in Settings to browse or delete
        </div>
      )}
    </div>
  )
}

// One group per vector store: badge (active/inactive) + a grid of index cards. Used by both the
// Ingestion page (read-only) and Indices → Explore (selectable + deletable).
export function KbOverview({
  overview,
  selectable = false,
  deletable = false,
  selectedIndex,
  onSelect,
  onDelete,
}: {
  overview: IndicesOverviewResponse | null
  selectable?: boolean
  deletable?: boolean
  selectedIndex?: string | null
  onSelect?: (store: string, index: string) => void
  onDelete?: (store: string, index: string) => void
}) {
  if (!overview) return null
  return (
    <div>
      {overview.stores.map((st) => (
        <div key={st.vector_store} className="mt-4 first:mt-0">
          <div className="flex items-center gap-2 pb-2 mb-3 border-b border-line">
            <span className="text-[0.82rem] font-semibold text-ink flex items-center gap-1.5">
              <DatabaseIcon size={14} className="text-signal" /> {st.vector_store}
            </span>
            <Pill tone={st.active ? 'ok' : 'default'}>{st.active ? 'active' : 'inactive'}</Pill>
          </div>
          {!st.available ? (
            <div className="text-warn text-sm flex items-center gap-1.5">
              <AlertTriangleIcon size={14} /> unavailable &mdash; {st.error || 'cannot reach this store'}
            </div>
          ) : !st.indices.length ? (
            <div className="text-ink-faint text-sm py-1">No indexes in this store yet.</div>
          ) : (
            <div className="grid gap-2.5" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
              {st.indices.map((ix) => (
                <OverviewCard
                  key={ix.index}
                  store={st.vector_store}
                  ix={ix}
                  active={st.active}
                  selectable={selectable}
                  deletable={deletable}
                  selected={selectedIndex === ix.index}
                  onSelect={onSelect}
                  onDelete={onDelete}
                />
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
