import { useState } from 'react'
import type { EvalOverall, EvalReport } from '../../api/types'
import { MetricBar } from '../../components/ScoreBar'
import { Panel } from '../../components/Panel'
import { Pill } from '../../components/Pill'
import { KCurveChart } from './charts'
import { METRIC_TITLE } from './metricDefs'
import { EvalCaseCard } from './EvalCaseCard'
import { systemTagText } from './systemTag'

const CUTOFFS = ['1', '3', '5', '10']

function CiBadge({ ci, asPct = true }: { ci?: [number, number]; asPct?: boolean }) {
  if (!ci || ci.length !== 2) return null
  const fmt = asPct ? (v: number) => `${(v * 100).toFixed(0)}%` : (v: number) => v.toFixed(3)
  return (
    <span className="text-xs text-ink-faint ml-1.5">
      95% CI {fmt(ci[0])}&ndash;{fmt(ci[1])}
    </span>
  )
}

function CurveTable({ m }: { m: EvalOverall }) {
  const cell = (obj: Record<string, number> | undefined, k: string) => `${((obj?.[k] ?? 0) * 100).toFixed(0)}%`
  const rows: [string, Record<string, number>][] = [
    ['Recall@k', m.recall_at],
    ['Precision@k', m.precision_at],
    ['nDCG@k', m.ndcg_at],
  ]
  return (
    <table className="w-full text-sm mt-2 border-collapse">
      <thead>
        <tr>
          <th className="text-left py-1.5 px-2 text-ink-dim font-semibold border-b border-line" />
          {CUTOFFS.map((k) => (
            <th key={k} className="text-left py-1.5 px-2 text-ink-dim font-semibold border-b border-line">
              {k}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, obj]) => (
          <tr key={label}>
            <td className="py-1.5 px-2 border-b border-line text-ink-dim">{label}</td>
            {CUTOFFS.map((k) => (
              <td key={k} className="py-1.5 px-2 border-b border-line font-mono">
                {cell(obj, k)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function BucketRow({ bucket, m }: { bucket: string; m: EvalOverall }) {
  return (
    <tr>
      <td className="py-1.5 px-2 border-b border-line">
        <Pill tone="soft">{bucket}</Pill>
      </td>
      <td className="py-1.5 px-2 border-b border-line">
        <MetricBar value={m.hit_rate ?? 0} />
      </td>
      <td className="py-1.5 px-2 border-b border-line">
        <MetricBar value={m.recall ?? 0} />
      </td>
      <td className="py-1.5 px-2 border-b border-line font-mono">{(m.mrr ?? 0).toFixed(3)}</td>
      <td className="py-1.5 px-2 border-b border-line font-mono">{(m.ndcg ?? 0).toFixed(3)}</td>
      <td className="py-1.5 px-2 border-b border-line font-mono">{(m.map ?? 0).toFixed(3)}</td>
      <td className="py-1.5 px-2 border-b border-line font-mono">{(m.latency_ms?.p50 ?? 0).toFixed(0)}ms</td>
      <td className="py-1.5 px-2 border-b border-line">{m.cases ?? '—'}</td>
    </tr>
  )
}

export function EvalReportView({ res }: { res: EvalReport }) {
  const overall = res.overall ?? ({} as EvalOverall)
  const byBucket = res.per_bucket ?? {}
  const nCases = overall.cases ?? res.cases?.length ?? '?'
  const nHit = res.cases?.filter((c) => c.matched > 0).length ?? 0
  const nMiss = (res.cases?.length ?? 0) - nHit
  const ci = overall.ci ?? {}
  const lat = overall.latency_ms ?? { p50: 0, p95: 0 }
  const [expandAll, setExpandAll] = useState<boolean | null>(null)
  const [casesOpen, setCasesOpen] = useState(false)

  return (
    <div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        <Panel title="Overall">
          <div className="flex flex-col gap-1">
            <div className="flex justify-between text-sm py-1" title={METRIC_TITLE.hit_rate}>
              <span className="text-ink-dim">Hit-rate</span>
              <span className="flex items-center">
                <MetricBar value={overall.hit_rate ?? 0} />
                <CiBadge ci={ci.hit_rate} />
              </span>
            </div>
            <div className="flex justify-between text-sm py-1" title={METRIC_TITLE.recall}>
              <span className="text-ink-dim">Recall (all)</span>
              <span className="flex items-center">
                <MetricBar value={overall.recall ?? 0} />
                <CiBadge ci={ci.recall} />
              </span>
            </div>
            <div className="flex justify-between text-sm py-1" title={METRIC_TITLE.ndcg}>
              <span className="text-ink-dim">nDCG@10</span>
              <span className="flex items-center">
                <MetricBar value={overall.ndcg ?? 0} />
                <CiBadge ci={ci.ndcg} />
              </span>
            </div>
            <div className="flex justify-between text-sm py-1" title={METRIC_TITLE.map}>
              <span className="text-ink-dim">MAP</span>
              <span className="flex items-center">
                <MetricBar value={overall.map ?? 0} />
                <CiBadge ci={ci.map} />
              </span>
            </div>
            <div className="flex justify-between text-sm py-1" title={METRIC_TITLE.mrr}>
              <span className="text-ink-dim">MRR</span>
              <span className="font-mono flex items-center">
                {(overall.mrr ?? 0).toFixed(3)}
                <CiBadge ci={ci.mrr} asPct={false} />
              </span>
            </div>
            <div className="flex justify-between text-sm py-1" title={METRIC_TITLE.latency}>
              <span className="text-ink-dim">Latency</span>
              <span className="font-mono">
                p50 {(lat.p50 ?? 0).toFixed(0)}ms &middot; p95 {(lat.p95 ?? 0).toFixed(0)}ms
              </span>
            </div>
            <div className="flex justify-between text-sm py-1">
              <span className="text-ink-dim">Cases</span>
              <span>{nCases}</span>
            </div>
          </div>
          <div className="text-[0.7rem] uppercase tracking-wide text-ink-dim font-bold mt-3 mb-1">Curves</div>
          <KCurveChart overall={overall} />
          <CurveTable m={overall} />
        </Panel>
        {Object.keys(byBucket).length > 0 && (
          <Panel title="Per bucket">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr>
                  {['Bucket', 'Hit-rate', 'Recall', 'MRR', 'nDCG', 'MAP', 'p50', 'n'].map((h) => (
                    <th key={h} className="text-left py-1.5 px-2 text-ink-dim font-semibold border-b border-line">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(byBucket).map(([b, m]) => (
                  <BucketRow key={b} bucket={b} m={m} />
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </div>

      {res.run_id && (
        <div className="text-xs text-ink-faint my-2">
          saved as run <code className="bg-surface-2 px-1.5 py-0.5 rounded">{res.run_id}</code>
        </div>
      )}
      {res.params && (
        <div className="text-xs text-ink-faint my-2">
          params:{' '}
          {Object.entries(res.params)
            .filter(([k, v]) => v != null && k !== 'system')
            .map(([pk, pv]) => `${pk}=${pv}`)
            .join(' · ')}
          {systemTagText(res.params.system)}
        </div>
      )}

      {res.cases?.length ? (
        <details open={casesOpen} onToggle={(e) => setCasesOpen((e.target as HTMLDetailsElement).open)} className="border border-line rounded-panel overflow-hidden mt-2">
          <summary className="px-4 py-3.5 cursor-pointer flex items-center gap-3">
            <span className="text-[1.05rem] font-bold text-ink">Case details</span>
            <span className="text-sm text-ink-dim">
              {res.cases.length} cases &middot; <span className="text-good">{nHit} hit</span> / <span className="text-bad">{nMiss} miss</span>
            </span>
            {!casesOpen && <span className="text-ink-faint text-xs ml-auto">click to expand</span>}
          </summary>
          <div className="flex items-center gap-2 px-4 py-2.5 border-t border-b border-line">
            <span className="text-xs text-ink-faint">Per-query breakdown &mdash; expected sources vs what retrieval actually returned.</span>
            <span className="flex-1" />
            <button className="text-xs text-ink-dim hover:text-ink px-2 py-1" onClick={() => setExpandAll(true)}>
              Expand all nodes
            </button>
            <button className="text-xs text-ink-dim hover:text-ink px-2 py-1" onClick={() => setExpandAll(false)}>
              Collapse all
            </button>
          </div>
          <div className="px-4 py-3.5">
            {res.cases.map((c, i) => (
              <EvalCaseCard key={i} c={c} forceOpen={expandAll ?? undefined} />
            ))}
          </div>
        </details>
      ) : null}
    </div>
  )
}
