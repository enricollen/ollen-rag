import type { EvalOverall } from '../../../api/types'
import { CHART_PALETTE } from '../../../lib/format'
import { pct, TipOverlay, useTip, yGrid } from './chartUtils'

const CUTOFFS = ['1', '3', '5', '10']

// Recall/precision/nDCG as k grows -- one line per metric across the reported cutoffs.
export function KCurveChart({ overall }: { overall: EvalOverall }) {
  const { tip, show, move, hide } = useTip()
  const W = 520
  const H = 240
  const mL = 34
  const mR = 78
  const mT = 14
  const mB = 30
  const plotW = W - mL - mR
  const plotH = H - mT - mB
  const xOf = (i: number) => mL + (i / (CUTOFFS.length - 1)) * plotW
  const yOf = (v: number) => mT + (1 - v) * plotH
  const series: { name: string; key: keyof EvalOverall; color: string }[] = [
    { name: 'Recall', key: 'recall_at', color: CHART_PALETTE[0] },
    { name: 'Precision', key: 'precision_at', color: CHART_PALETTE[1] },
    { name: 'nDCG', key: 'ndcg_at', color: CHART_PALETTE[2] },
  ]

  return (
    <div className="relative mt-1">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} className="bg-surface-2/60 border border-line rounded-control">
        {yGrid(mL, W - mR, yOf, [0, 0.25, 0.5, 0.75, 1])}
        {CUTOFFS.map((k, i) => (
          <text key={k} x={xOf(i)} y={H - mB + 16} textAnchor="middle" fontSize={10} fill="var(--color-ink-faint)">
            k={k}
          </text>
        ))}
        {series.map((s) => {
          const obj = (overall[s.key] as Record<string, number>) ?? {}
          const pts = CUTOFFS.map((k, i) => ({ x: xOf(i), y: yOf(obj[k] ?? 0), v: obj[k] ?? 0, k }))
          const last = pts[pts.length - 1]
          return (
            <g key={s.name}>
              <polyline points={pts.map((p) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} />
              {pts.map((p) => (
                <circle
                  key={p.k}
                  cx={p.x}
                  cy={p.y}
                  r={4}
                  fill={s.color}
                  stroke="var(--color-surface-2)"
                  strokeWidth={2}
                  className="cursor-pointer"
                  onMouseEnter={(e) => show(e, `${s.name}@${p.k}: ${pct(p.v)}`)}
                  onMouseMove={move}
                  onMouseLeave={hide}
                />
              ))}
              <text x={last.x + 8} y={last.y + 3} fontSize={11} fill={s.color}>
                {s.name}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="flex gap-4 mt-2">
        {series.map((s) => (
          <span key={s.name} className="inline-flex items-center gap-1.5 text-xs text-ink-dim">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
      <TipOverlay tip={tip} />
    </div>
  )
}
