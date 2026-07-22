// XHR-based upload (not fetch) so we get upload.onprogress for the progress bar. One POST per
// file; caller drives the serial batch loop (see lib/ingestBatch.ts).
import { endpoints } from './client'
import type { IngestJobStatus } from './types'

export interface IngestOneParams {
  file: File
  strategy?: string | null
  indexName?: string | null
  embProvider?: string | null
  embModel?: string | null
  chunkParams?: Record<string, number>
  metadata?: Record<string, unknown>
  enrich?: boolean
  onProgress?: (pct: number) => void
}

export function ingestOne(params: IngestOneParams): Promise<{ job_id: string; status: string }> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', params.file)
    if (params.strategy) form.append('strategy', params.strategy)
    if (params.indexName) form.append('index_name', params.indexName)
    if (params.embProvider) form.append('embedding_provider', params.embProvider)
    if (params.embModel) form.append('embedding_model', params.embModel)
    if (params.chunkParams && Object.keys(params.chunkParams).length) {
      form.append('chunk_params', JSON.stringify(params.chunkParams))
    }
    if (params.metadata && Object.keys(params.metadata).length) {
      form.append('metadata', JSON.stringify(params.metadata))
    }
    if (params.enrich) form.append('enrich_keywords', 'true')

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/v1/ingest')
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && params.onProgress) {
        params.onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }
    xhr.onload = () => {
      let body: unknown
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        body = xhr.responseText
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as { job_id: string; status: string })
      else reject(Object.assign(new Error(`HTTP ${xhr.status}`), { body }))
    }
    // xhr.onerror fires for connection refused / dns / aborted — not for http 4xx/5xx
    xhr.onerror = () =>
      reject(new Error('Network error — could not reach /api/v1/ingest (is the service up on :8000?)'))
    xhr.ontimeout = () => reject(new Error('Upload timed out'))
    xhr.timeout = 10 * 60 * 1000
    xhr.send(form)
  })
}

// Polls a job until it reaches a terminal state, invoking onTick on every poll (incl. the terminal one).
export function pollJob(jobId: string, onTick: (status: IngestJobStatus) => void): Promise<IngestJobStatus | null> {
  return new Promise((resolve) => {
    const timer = setInterval(async () => {
      try {
        const res = await endpoints.ingestStatus(jobId)
        onTick(res)
        if (res.status === 'completed' || res.status === 'failed') {
          clearInterval(timer)
          resolve(res)
        }
      } catch {
        clearInterval(timer)
        resolve(null)
      }
    }, 1500)
  })
}
