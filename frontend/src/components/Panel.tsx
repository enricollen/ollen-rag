import type { ReactNode } from 'react'

interface PanelProps {
  title?: ReactNode
  subtitle?: ReactNode
  badge?: ReactNode
  accent?: boolean
  dim?: boolean
  className?: string
  children: ReactNode
}

// The base "instrument" card used throughout the console. `accent` gives it the amber
// control-panel treatment (used for the top settings gate); `dim` fades an inactive block.
export function Panel({ title, subtitle, badge, accent, dim, className = '', children }: PanelProps) {
  return (
    <div
      className={`glass-panel p-5 mb-4 transition-opacity ${dim ? 'opacity-45' : ''} ${className}`}
      style={accent ? { borderColor: 'color-mix(in srgb, var(--color-accent) 45%, transparent)' } : undefined}
    >
      {(title || badge) && (
        <div className="flex items-center gap-2 mb-3">
          {title && <h2 className="text-sm font-semibold text-ink m-0">{title}</h2>}
          {badge}
        </div>
      )}
      {subtitle && <p className="text-xs text-ink-dim mt-0 mb-3 -mt-1">{subtitle}</p>}
      {children}
    </div>
  )
}
