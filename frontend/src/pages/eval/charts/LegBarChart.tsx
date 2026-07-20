import type { EvalLegOverall } from '../../../api/types'
import { CHART_PALETTE } from '../../../lib/format'
import { AXIS, BAD, GOOD, signed, TEXT, TipOverlay, useTip, yGrid } from './chartUtils'

const LEGS = ['bm25', 'dense', 'hybrid', 'reranked'] as const

// One metric across the four retrieval legs, with the reranked bar annotated by lift.
export function LegBarChart({
  perLeg,
  rerankLift,
  metric,
}: {
  perLeg: Record<string, { overall: EvalLegOverall }>
  rerankLift: Record<string, number>
  metric: string
}) {
  const { tip, show, move, hide } = useTip()
  const W = 440
  const H = 220
  const mL = 34
  const mR = 14
  const mT = 22
  const mB = 28
  const plotW = W - mL - mR
  const plotH = H - mT - mB
  const yOf = (v: number) => mT + (1 - v) * plotH
  const slot = plotW / LEGS.length
  const barW = slot * 0.56
  const color = CHART_PALETTE[0]
  const lift = rerankLift?.[metric] ?? 0
  const liftCol = lift > 0 ? GOOD : lift < 0 ? BAD : AXIS

  return (
    <div className="relative mt-1">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} className="bg-surface-2/60 border border-line rounded-control">
        {yGrid(mL, W - mR, yOf, [0, 0.25, 0.5, 0.75, 1])}
        <text x={W - mR} y={mT - 8} textAnchor="end" fontSize={11} fill={liftCol}>
          rerank lift {signed(lift)}
        </text>
        {LEGS.map((leg, i) => {
          const v = (perLeg?.[leg]?.overall as unknown as Record<string, number>)?.[metric] ?? 0
          const x = mL + i * slot + (slot - barW) / 2
          const y = yOf(v)
          const h = Math.max(1, H - mB - y)
          return (
            <g key={leg}>
              <rect
                x={x}
                y={y}
                width={barW}
                height={h}
                rx={4}
                fill={color}
                fillOpacity={leg === 'reranked' ? 1 : 0.7}
                className="cursor-pointer"
                onMouseEnter={(e) => show(e, `${leg} ${metric} ${v.toFixed(3)}`)}
                onMouseMove={move}
                onMouseLeave={hide}
              />
              <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize={10} fill={TEXT}>
                {v.toFixed(2)}
              </text>
              <text x={x + barW / 2} y={H - mB + 14} textAnchor="middle" fontSize={10} fill={AXIS}>
                {leg}
              </text>
            </g>
          )
        })}
      </svg>
      <TipOverlay tip={tip} />
    </div>
  )
}
