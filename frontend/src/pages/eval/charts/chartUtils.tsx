import { useState } from 'react'

// Shared visual language for every hand-rolled eval SVG chart -- kept in one place so the three
// chart components (KCurveChart/DeltaChart/LegBarChart) stay visually consistent. References the
// live theme tokens (not literal hex) so charts stay legible across the dark/light toggle.
export const AXIS = 'var(--color-ink-faint)'
export const TEXT = 'var(--color-ink-dim)'
export const GOOD = 'var(--color-good)'
export const BAD = 'var(--color-bad)'
export const GRID = 'var(--color-line)'

export const pct = (v: number | undefined) => `${Math.round((v ?? 0) * 100)}%`
export const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(4)}`

export interface Tip {
  x: number
  y: number
  text: string
}

// Cursor-following tooltip state for hovering individual marks (points/bars) in an SVG chart.
export function useTip() {
  const [tip, setTip] = useState<Tip | null>(null)
  return {
    tip,
    show: (e: React.MouseEvent, text: string) => setTip({ x: e.clientX, y: e.clientY, text }),
    move: (e: React.MouseEvent) => setTip((t) => (t ? { ...t, x: e.clientX, y: e.clientY } : t)),
    hide: () => setTip(null),
  }
}

export function TipOverlay({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  return (
    <div
      className="fixed pointer-events-none bg-surface border border-line rounded-control px-2.5 py-1.5 text-sm z-10 whitespace-nowrap"
      style={{ left: tip.x + 12, top: tip.y + 12 }}
    >
      {tip.text}
    </div>
  )
}

// Horizontal gridlines + percentage axis labels at the given fractions (e.g. [0, .25, .5, .75, 1]).
export function yGrid(x0: number, x1: number, yOf: (f: number) => number, fractions: number[]) {
  return fractions.map((f) => {
    const y = yOf(f)
    return (
      <g key={f}>
        <line x1={x0} y1={y} x2={x1} y2={y} stroke={GRID} strokeWidth={1} />
        <text x={x0 - 6} y={y + 3} textAnchor="end" fontSize={10} fill={AXIS}>
          {Math.round(f * 100)}%
        </text>
      </g>
    )
  })
}
