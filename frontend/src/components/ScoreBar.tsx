// Visual score bar for retrieval/rerank scores, which are unbounded -- normalized onto a rough
// [-5, 10] display scale (matches the old console's heuristic) purely for a relative sense of magnitude.
export function ScoreBar({ score }: { score: number }) {
  const normalized = Math.max(0, Math.min(1, (score + 5) / 15))
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden flex-1 max-w-[160px]">
        <div className="h-full rounded-full bg-gradient-to-r from-accent to-signal" style={{ width: `${Math.round(normalized * 100)}%` }} />
      </div>
      <span className="font-mono text-xs text-signal flex-shrink-0">{score.toFixed(3)}</span>
    </div>
  )
}

// Percentage-scale metric bar (eval hit-rate/recall/etc, already 0..1), color-graded by value.
export function MetricBar({ value }: { value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100)
  const color = value >= 0.8 ? 'var(--color-good)' : value >= 0.5 ? 'var(--color-warn)' : 'var(--color-bad)'
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden flex-1 max-w-[160px]">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="font-mono text-xs text-ink-dim">{(value * 100).toFixed(1)}%</span>
    </div>
  )
}
