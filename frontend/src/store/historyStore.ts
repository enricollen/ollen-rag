// Session-local persisted state (survives page nav, not server-backed): ingestion job history,
// remembered bucket names, and the Q&A thread. Mirrors ui/lib.js's localStorage helpers, backed by
// zustand's persist middleware under the same storage key so an existing browser's history carries over.
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { IngestJobStatus } from '../api/types'
import type { QuerySource } from '../api/types'

export interface JobHistoryEntry {
  job_id: string
  status: string
  file_name?: string
  strategy?: string | null
  bucket?: string
  embedding_model?: string
  detail?: string | null
  result?: IngestJobStatus['result']
  progress?: number
  stage?: string
}

export interface QaHistoryEntry {
  query: string
  answer: string
  sources: QuerySource[]
}

interface HistoryState {
  jobs: JobHistoryEntry[]
  buckets: string[]
  qa: QaHistoryEntry[]
  addJob: (job: JobHistoryEntry) => void
  updateJob: (jobId: string, patch: Partial<JobHistoryEntry>) => void
  clearJobs: () => void
  rememberBucket: (bucket: string) => void
  addQa: (entry: QaHistoryEntry) => void
  clearQa: () => void
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set) => ({
      jobs: [],
      buckets: [],
      qa: [],
      addJob: (job) => set((s) => ({ jobs: [job, ...s.jobs].slice(0, 30) })),
      updateJob: (jobId, patch) =>
        set((s) => ({ jobs: s.jobs.map((j) => (j.job_id === jobId ? { ...j, ...patch } : j)) })),
      clearJobs: () => set({ jobs: [] }),
      rememberBucket: (bucket) =>
        set((s) => {
          if (!bucket) return s
          const buckets = [...new Set([...s.buckets, bucket])].slice(-20)
          return { buckets }
        }),
      addQa: (entry) => set((s) => ({ qa: [entry, ...s.qa].slice(0, 20) })),
      clearQa: () => set({ qa: [] }),
    }),
    { name: 'ollen_rag_ui_state_v1' },
  ),
)
