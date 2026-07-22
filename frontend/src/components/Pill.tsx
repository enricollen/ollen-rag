import type { ReactNode } from 'react'

type PillTone = 'default' | 'ok' | 'warn' | 'bad' | 'soft'

const TONES: Record<PillTone, string> = {
  default: 'border-line text-ink-dim',
  ok: 'border-good/50 text-good',
  warn: 'border-warn/50 text-warn',
  bad: 'border-bad/50 text-bad',
  soft: 'border-accent/30 text-ink-dim bg-accent-soft',
}

export function Pill({
  tone = 'default',
  children,
  className = '',
  title,
  onClick,
}: {
  tone?: PillTone
  children: ReactNode
  className?: string
  title?: string
  onClick?: (e: React.MouseEvent) => void
}) {
  return (
    <span
      title={title}
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded-full border bg-surface-2/60 px-2.5 py-0.5 text-[0.7rem] leading-normal whitespace-nowrap ${TONES[tone]} ${onClick ? 'cursor-pointer hover:border-accent/50' : ''} ${className}`}
    >
      {children}
    </span>
  )
}
