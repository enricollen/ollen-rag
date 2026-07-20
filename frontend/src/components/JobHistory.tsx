import { useEffect } from 'react'
import { pollJob } from '../api/ingest'
import { useHistoryStore, type JobHistoryEntry } from '../store/historyStore'
import { EmptyState, ProgressBar } from './Misc'

const STATUS_CLS: Record<string, string> = {
  pending: 'text-warn bg-warn/15',
  running: 'text-warn bg-warn/15',
  completed: 'text-good bg-good/15',
  failed: 'text-bad bg-bad/15',
}

function JobCard({ job }: { job: JobHistoryEntry }) {
  const status = job.status || 'pending'
  const running = status === 'pending' || status === 'running'
  const detail = job.result
    ? job.result.skipped_duplicate
      ? (
        <>
          duplicate skipped &mdash; same content already indexed as <strong>{job.result.duplicate_of || ''}</strong> in{' '}
          <code className="bg-surface-2 px-1 rounded">{job.result.index}</code>
        </>
      )
      : (
        <>
          {job.result.num_documents} doc &rarr; {job.result.num_nodes} chunks in <code className="bg-surface-2 px-1 rounded">{job.result.index}</code>
        </>
      )
    : job.detail || `job id: ${job.job_id}`

  return (
    <div className="bg-surface-2/60 border border-line rounded-panel px-4 py-3.5 mb-2.5">
      <div className="flex justify-between items-center">
        <div>
          <strong className="text-ink text-sm">{job.file_name || 'document'}</strong>{' '}
          <span className="text-ink-faint text-xs">
            {job.strategy || ''}
            {job.bucket ? ` · bucket: ${job.bucket}` : ''}
            {job.embedding_model ? ` · ${job.embedding_model}` : ''}
          </span>
        </div>
        <span className={`text-[0.72rem] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wide ${STATUS_CLS[status] ?? ''}`}>{status}</span>
      </div>
      <div className="text-ink-dim text-[0.76rem] mt-1.5">{detail}</div>
      {running && (
        <div className="mt-2">
          <ProgressBar pct={job.progress || 0} />
          <div className="text-ink-faint text-xs mt-1">
            {job.stage || 'starting'} · {job.progress || 0}%
          </div>
        </div>
      )}
    </div>
  )
}

// Renders the session's job history, re-polling any still-running jobs to keep their cards live.
export function JobHistory() {
  const jobs = useHistoryStore((s) => s.jobs)
  const updateJob = useHistoryStore((s) => s.updateJob)

  useEffect(() => {
    const running = jobs.filter((j) => j.status === 'pending' || j.status === 'running')
    running.forEach((j) => {
      pollJob(j.job_id, (res) => updateJob(j.job_id, res))
    })
    // Intentionally runs once per mount -- polling continues job-by-job via pollJob's own timer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!jobs.length) return <EmptyState>No ingestion jobs yet this session.</EmptyState>
  return (
    <div>
      {jobs.map((j) => (
        <JobCard key={j.job_id} job={j} />
      ))}
    </div>
  )
}
