import { useState } from 'react'
import type { IndexInfo } from '../api/types'
import { chunkingSummary } from '../lib/format'
import { Select } from './Field'
import { FileTextIcon } from './icons'

// Shared "locked configuration" readout for an index: chunking + embedding model + dim + docs +
// buckets, plus an optional bucket -> files picker. `verb` names what runs against it (e.g.
// "Queries", "Eval cases") so the lock note reads naturally.
export function IndexInfoPanel({
  info,
  verb = 'Queries',
  showBucketFiles = true,
  onBucketSelect,
}: {
  info: IndexInfo
  verb?: string
  showBucketFiles?: boolean
  onBucketSelect?: (bucket: string) => void
}) {
  const emb = info.embedding_provider ? `${info.embedding_provider}/${info.embedding_model}` : 'unrecorded (legacy index)'
  const bucketNames = Object.keys(info.bucket_files || {})
  const [selectedBucket, setSelectedBucket] = useState('')
  const files = info.bucket_files?.[selectedBucket] ?? []

  return (
    <div className="mb-4 p-4 border border-line border-l-2 border-l-accent rounded-panel bg-surface-2/50">
      <div className="font-bold text-sm mb-2 text-ink">
        <code className="bg-surface-2 px-1.5 py-0.5 rounded font-mono text-[0.82rem]">{info.index}</code>{' '}
        <span className="text-ink-dim font-normal">&mdash; locked configuration</span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm text-ink-dim">
        <span>
          <strong className="text-ink-dim">chunking</strong> {chunkingSummary(info.chunking)}
        </span>
        <span>
          <strong className="text-ink-dim">embedding</strong> {emb}
        </span>
        <span>
          <strong className="text-ink-dim">dim</strong> {info.dim ?? '?'}
        </span>
        <span>
          <strong className="text-ink-dim">docs</strong> {info.docs_count ?? '?'}
        </span>
        <span>
          <strong className="text-ink-dim">buckets</strong> {info.buckets.join(', ') || '—'}
        </span>
      </div>
      {showBucketFiles && bucketNames.length > 0 && (
        <div className="mt-3">
          <label className="block text-xs text-ink-faint mb-1">
            Buckets &amp; documents in this index
            <Select
              className="mt-1"
              value={selectedBucket}
              onChange={(e) => {
                setSelectedBucket(e.target.value)
                onBucketSelect?.(e.target.value)
              }}
            >
              <option value="">(choose a bucket)</option>
              {bucketNames.map((b) => (
                <option key={b} value={b}>
                  {b} ({info.bucket_files[b]?.length ?? 0})
                </option>
              ))}
            </Select>
          </label>
          {selectedBucket && (
            <ul className="mt-1.5 list-none m-0 p-0 flex flex-col gap-0.5 text-xs text-ink-dim">
              {files.length ? (
                files.map((f) => (
                  <li key={f} className="flex items-center gap-1.5">
                    <FileTextIcon size={12} className="text-ink-faint" /> {f}
                  </li>
                ))
              ) : (
                <li className="text-ink-faint">empty</li>
              )}
            </ul>
          )}
        </div>
      )}
      <div className="mt-2 text-[0.8rem] text-signal">
        {verb} run against this index use its recorded embedding model &mdash; no model mixing.
      </div>
    </div>
  )
}
