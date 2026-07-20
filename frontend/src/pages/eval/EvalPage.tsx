import { useCallback, useEffect, useState } from 'react'
import { endpoints } from '../../api/client'
import type { EvalRunSummary } from '../../api/types'
import { BarChartIcon } from '../../components/icons'
import { PageHeader } from '../../components/PageHeader'
import { EvalRunForm } from './EvalRunForm'
import { EvalRunHistory } from './EvalRunHistory'
import { MetricsLegend } from './MetricsLegend'

export function EvalPage() {
  const [runs, setRuns] = useState<EvalRunSummary[]>([])

  const loadRuns = useCallback(async () => {
    try {
      const { runs: r } = await endpoints.evalRuns()
      setRuns(r)
    } catch {
      // history is best-effort
    }
  }, [])

  useEffect(() => {
    loadRuns()
  }, [loadRuns])

  return (
    <div>
      <PageHeader icon={BarChartIcon} title="Retrieval Eval">
        Run the golden-dataset eval harness (<code className="bg-surface-2 px-1.5 py-0.5 rounded">POST /api/v1/eval/retrieval</code>).
        Retrieval is <strong>bucket-agnostic</strong> &mdash; it searches the whole index, so make sure the chosen index actually contains
        the dataset's documents (a mismatch just returns 0 hits). A case's optional{' '}
        <code className="bg-surface-2 px-1.5 py-0.5 rounded">bucket</code> only labels its metrics. Metrics: hit-rate, recall@k,
        precision@k, MRR, nDCG, MAP &mdash; with per-k curves, latency, and 95% bootstrap CIs, overall and per label.
      </PageHeader>

      <MetricsLegend />
      <EvalRunForm onRunSaved={loadRuns} />
      <EvalRunHistory runs={runs} onRefresh={loadRuns} />
    </div>
  )
}
