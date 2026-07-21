// Same-origin fetch wrapper (mirrors the old ui/lib.js api()) -- throws ApiError with the parsed
// {error_code, detail} body attached so callers can render a real message instead of "Failed to fetch".
export class ApiError extends Error {
  status: number
  body: unknown
  constructor(status: number, body: unknown) {
    super(`HTTP ${status}`)
    this.status = status
    this.body = body
  }
}

export async function api<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(path, options)
  const text = await res.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  if (!res.ok) throw new ApiError(res.status, body)
  return body as T
}

function postJson<T>(path: string, payload: unknown): Promise<T> {
  return api<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

// Renders whatever shape a failed api() call produced into a short string.
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    const body = err.body as { detail?: unknown } | string | null
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = body.detail
      if (typeof detail === 'string') return detail
      if (detail != null) return JSON.stringify(detail)
    }
    if (typeof body === 'string' && body) return body
    return err.message
  }
  if (err instanceof Error) return err.message
  return 'unexpected error'
}

import type {
  CompareResponse,
  ConfigResponse,
  EvalLegReport,
  EvalParams,
  EvalReport,
  EvalRunsResponse,
  IndexDocumentsResponse,
  IndexInfo,
  IndexListEntry,
  IndexVectorsResponse,
  IndicesOverviewResponse,
  IngestJobStatus,
  OnboardingStatus,
  OnboardingTestRequest,
  OnboardingTestResponse,
  QueryParams,
  QueryResponse,
  RetrieveParams,
  RetrieveResponse,
  SettingsDump,
} from './types'

export const endpoints = {
  health: () => api<{ status: string }>('/health'),
  ready: () => api<{ status: string }>('/ready'),
  strategies: () => api<{ strategies: string[] }>('/api/v1/strategies'),

  config: () => api<ConfigResponse>('/api/v1/config'),
  settings: () => api<SettingsDump>('/api/v1/settings'),
  saveSettings: (changes: Record<string, unknown>) =>
    postJson<{ restarting: boolean; restart_mode: string; applied_live: boolean }>('/api/v1/settings', changes),

  onboardingStatus: () => api<OnboardingStatus>('/api/v1/onboarding/status'),
  onboardingTest: (req: OnboardingTestRequest) =>
    postJson<OnboardingTestResponse>('/api/v1/onboarding/test', req),

  opensearchStatus: () => api<{ reachable: boolean }>('/api/v1/infra/opensearch/status'),
  qdrantStatus: () => api<{ reachable: boolean }>('/api/v1/infra/qdrant/status'),

  indices: () => api<{ indices: IndexListEntry[] }>('/api/v1/indices'),
  indicesOverview: () => api<IndicesOverviewResponse>('/api/v1/indices/overview'),
  indexInfo: (name: string) => api<IndexInfo>(`/api/v1/indices/${encodeURIComponent(name)}/info`),
  indexBuckets: (name: string) =>
    api<{ buckets: string[] }>(`/api/v1/indices/${encodeURIComponent(name)}/buckets`),
  indexVectors: (name: string, limit = 2000) =>
    api<IndexVectorsResponse>(`/api/v1/indices/${encodeURIComponent(name)}/vectors?limit=${limit}`),
  indexDocuments: (
    name: string,
    params: { offset: number; limit: number; bucket?: string | null; unbucketed?: boolean; file_name?: string | null },
  ) => {
    const qs = new URLSearchParams({ offset: String(params.offset), limit: String(params.limit) })
    if (params.bucket) qs.set('bucket', params.bucket)
    if (params.unbucketed) qs.set('unbucketed', 'true')
    if (params.file_name) qs.set('file_name', params.file_name)
    return api<IndexDocumentsResponse>(`/api/v1/indices/${encodeURIComponent(name)}/documents?${qs}`)
  },
  deleteIndex: (name: string) => api<{ deleted: string }>(`/api/v1/indices/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  deleteBucket: (name: string, bucket: string) =>
    api<{ deleted: number; bucket: string; index: string }>(
      `/api/v1/indices/${encodeURIComponent(name)}/buckets/${encodeURIComponent(bucket)}`,
      { method: 'DELETE' },
    ),

  retrieve: (params: RetrieveParams) => postJson<RetrieveResponse>('/api/v1/retrieve', params),
  query: (params: QueryParams) => postJson<QueryResponse>('/api/v1/query', params),

  ingestStatus: (jobId: string) => api<IngestJobStatus>(`/api/v1/ingest/${jobId}`),

  evalRetrieval: (params: EvalParams) => postJson<EvalReport | EvalLegReport>('/api/v1/eval/retrieval', params),
  evalRuns: () => api<EvalRunsResponse>('/api/v1/eval/runs'),
  evalCompare: (a: string, b: string) => postJson<CompareResponse>('/api/v1/eval/compare', { a, b }),
}
