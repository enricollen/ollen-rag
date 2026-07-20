import type { CompareResponse } from '../../../api/types'
import { AXIS, BAD, GOOD, signed, TEXT, TipOverlay, useTip } from './chartUtils'

const METRICS = ['hit_rate', 'recall', 'mrr', 'ndcg', 'map']

// Paired A/B deltas (B-A) as diverging bars around a zero line, with 95% CI whiskers.
export function DeltaChart({ cmp }: { cmp: CompareResponse }) {
  const { tip, show, move, hide } = useTip()
  const rows = METRICS.filter((m) => cmp.metrics?.[m])
  const W = 560
  const rowH = 30
  const mL = 68
  const mR = 54
  const mT = 12
  const mB = 22
  const H = mT + rows.length * rowH + mB
  const plotW = W - mL - mR
  const zeroX = mL + plotW / 2
  const D = Math.max(
    0.1,
    ...rows.flatMap((m) => {
      const d = cmp.metrics[m]
      return [Math.abs(d.delta), Math.abs(d.ci[0]), Math.abs(d.ci[1])]
    }),
  )
  const xOf = (v: number) => zeroX + (v / D) * (plotW / 2)

  return (
    <div className="relative mt-1">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} className="bg-surface-2/60 border border-line rounded-control">
        <line x1={zeroX} y1={mT} x2={zeroX} y2={H - mB} stroke={AXIS} strokeWidth={1.5} />
        <text x={zeroX} y={H - mB + 14} textAnchor="middle" fontSize={10} fill={AXIS}>
          0
        </text>
        <text x={mL} y={H - mB + 14} textAnchor="middle" fontSize={10} fill={AXIS}>
          -{D.toFixed(2)}
        </text>
        <text x={W - mR} y={H - mB + 14} textAnchor="middle" fontSize={10} fill={AXIS}>
          +{D.toFixed(2)}
        </text>
        {rows.map((m, i) => {
          const d = cmp.metrics[m]
          const cy = mT + i * rowH + rowH / 2
          const color = d.delta > 0 ? GOOD : d.delta < 0 ? BAD : AXIS
          const x0 = Math.min(zeroX, xOf(d.delta))
          const x1 = Math.max(zeroX, xOf(d.delta))
          const barW = Math.max(1, x1 - x0)
          const labelX = Math.max(x1, xOf(d.ci[1])) + 6
          const tip_ = `${m} Δ ${signed(d.delta)} · 95% CI [${d.ci[0].toFixed(3)}, ${d.ci[1].toFixed(3)}]${d.significant ? ' · significant' : ''}`
          return (
            <g key={m}>
              <text x={mL - 8} y={cy + 3} textAnchor="end" fontSize={11} fill={TEXT}>
                {m}
                {d.significant ? ' ●' : ''}
              </text>
              <line x1={xOf(d.ci[0])} y1={cy} x2={xOf(d.ci[1])} y2={cy} stroke={TEXT} strokeWidth={1.5} />
              <line x1={xOf(d.ci[0])} y1={cy - 4} x2={xOf(d.ci[0])} y2={cy + 4} stroke={TEXT} strokeWidth={1.5} />
              <line x1={xOf(d.ci[1])} y1={cy - 4} x2={xOf(d.ci[1])} y2={cy + 4} stroke={TEXT} strokeWidth={1.5} />
              <rect
                x={x0}
                y={cy - 7}
                width={barW}
                height={14}
                rx={3}
                fill={color}
                fillOpacity={0.85}
                className="cursor-pointer"
                onMouseEnter={(e) => show(e, tip_)}
                onMouseMove={move}
                onMouseLeave={hide}
              />
              <text x={labelX} y={cy + 3} fontSize={10} fill={TEXT}>
                {signed(d.delta)}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="flex gap-4 mt-2 text-xs text-ink-dim">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: GOOD }} /> B better
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: BAD }} /> A better
        </span>
        <span>&#9679; = significant (CI excludes 0)</span>
      </div>
      <TipOverlay tip={tip} />
    </div>
  )
}
