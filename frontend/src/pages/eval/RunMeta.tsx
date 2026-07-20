import type { EvalRunSummary } from '../../api/types'
import { systemTagShort } from './runFormat'

// Headline metrics + system tag for a picked run, shown under its A/B select so the two sides are
// legible at a glance without opening the compare. Falls back quietly for older runs.
export function RunMeta({ run }: { run?: EvalRunSummary }) {
  if (!run) return <span className="text-ink-faint text-xs">no run selected</span>
  const o = run.overall ?? {}
  const bits: string[] = []
  if (o.ndcg != null) bits.push(`nDCG ${o.ndcg.toFixed(3)}`)
  if (o.recall != null) bits.push(`recall ${o.recall.toFixed(3)}`)
  if (o.hit_rate != null) bits.push(`hit ${(o.hit_rate * 100).toFixed(0)}%`)
  const sys = systemTagShort(run).replace(/^ · /, '')
  return (
    <div>
      <span className="font-mono text-xs text-ink-dim">{bits.join(' · ') || '—'}</span>
      {sys && <div className="text-xs text-ink-faint mt-0.5">{sys}</div>}
    </div>
  )
}
