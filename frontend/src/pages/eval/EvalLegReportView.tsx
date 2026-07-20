import { useState } from 'react'
import type { EvalLegReport } from '../../api/types'
import { MetricBar } from '../../components/ScoreBar'
import { Panel } from '../../components/Panel'
import { LegBarChart } from './charts'

const LEGS = ['bm25', 'dense', 'hybrid', 'reranked'] as const
const LIFT_METRICS = ['ndcg', 'recall', 'mrr', 'map', 'hit_rate']
const CHART_METRICS = ['ndcg', 'recall', 'mrr', 'map', 'hit_rate']

export function EvalLegReportView({ res }: { res: EvalLegReport }) {
  const [chartMetric, setChartMetric] = useState('ndcg')
  const lift = res.rerank_lift ?? {}

  return (
    <Panel
      title="Per-leg attribution"
      subtitle={
        <>
          Each retrieval leg scored on its own. <strong>Rerank lift</strong> = reranked &minus; hybrid: what the cross-encoder
          adds. Per-leg latency is not measured (one debug call serves all legs).
        </>
      }
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm text-ink-dim">Chart metric</span>
        <select
          value={chartMetric}
          onChange={(e) => setChartMetric(e.target.value)}
          className="max-w-[9rem] bg-surface-2 border border-line rounded-control px-2 py-1 text-sm text-ink"
        >
          {CHART_METRICS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
      <LegBarChart perLeg={res.per_leg} rerankLift={lift} metric={chartMetric} />
      <table className="w-full text-sm mt-3 border-collapse">
        <thead>
          <tr>
            {['Leg', 'Hit-rate', 'Recall', 'MRR', 'nDCG', 'MAP'].map((h) => (
              <th key={h} className="text-left py-1.5 px-2 text-ink-dim font-semibold border-b border-line">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {LEGS.map((leg) => {
            const o = res.per_leg?.[leg]?.overall ?? ({} as never)
            return (
              <tr key={leg}>
                <td className="py-1.5 px-2 border-b border-line text-ink-dim">{leg}</td>
                <td className="py-1.5 px-2 border-b border-line">
                  <MetricBar value={o.hit_rate ?? 0} />
                </td>
                <td className="py-1.5 px-2 border-b border-line font-mono">{(o.recall ?? 0).toFixed(3)}</td>
                <td className="py-1.5 px-2 border-b border-line font-mono">{(o.mrr ?? 0).toFixed(3)}</td>
                <td className="py-1.5 px-2 border-b border-line font-mono">{(o.ndcg ?? 0).toFixed(3)}</td>
                <td className="py-1.5 px-2 border-b border-line font-mono">{(o.map ?? 0).toFixed(3)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="text-[0.7rem] uppercase tracking-wide text-ink-dim font-bold mt-3 mb-1.5">Rerank lift</div>
      <div className="flex flex-wrap gap-2">
        {LIFT_METRICS.map((m) => {
          const v = lift[m] ?? 0
          const cls = v > 0 ? 'text-good border-good' : v < 0 ? 'text-bad border-bad' : 'text-ink-dim border-line'
          return (
            <span key={m} className={`inline-flex items-center px-2.5 py-1 rounded-full border text-xs font-mono ${cls}`}>
              {m} {v >= 0 ? '+' : ''}
              {v.toFixed(4)}
            </span>
          )
        })}
      </div>
    </Panel>
  )
}
