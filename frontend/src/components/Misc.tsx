// Small stateless building blocks that don't earn their own file: status dot, spinner, empty
// state, and the thin animated progress bar used for uploads/ingestion.
import { motion } from 'motion/react'

export function StatusDot({ status }: { status: 'ok' | 'bad' | 'pending' }) {
  const color = status === 'ok' ? 'var(--color-good)' : status === 'bad' ? 'var(--color-bad)' : 'var(--color-ink-faint)'
  return (
    <span
      className="inline-block w-2 h-2 rounded-full flex-shrink-0"
      style={{ background: color, boxShadow: status !== 'pending' ? `0 0 8px ${color}` : undefined }}
    />
  )
}

export function Spinner({ className = '' }: { className?: string }) {
  return (
    <span
      className={`inline-block w-4 h-4 rounded-full border-2 border-line-strong border-t-signal animate-spin ${className}`}
    />
  )
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="text-center py-10 px-4 text-ink-faint text-sm">{children}</div>
}

export function ProgressBar({ pct }: { pct: number }) {
  return (
    <div className="h-1.5 rounded-full bg-line overflow-hidden">
      <motion.div
        className="h-full rounded-full"
        style={{ background: 'linear-gradient(90deg, var(--color-accent), var(--color-signal))', boxShadow: 'var(--glow-signal)' }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
        transition={{ ease: 'easeOut', duration: 0.25 }}
      />
    </div>
  )
}
