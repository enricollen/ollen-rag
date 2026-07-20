import type { ReactNode } from 'react'
import type { IconProps } from './icons'

// Consistent "icon chip + title + description" header used at the top of every top-level page,
// mirroring the icon language already used per-route in the Sidebar.
export function PageHeader({
  icon: PageIcon,
  title,
  children,
}: {
  icon: (props: IconProps) => ReactNode
  title: string
  children?: ReactNode
}) {
  return (
    <div className="flex items-start gap-3 mb-4">
      <div
        className="w-9 h-9 rounded-[10px] flex items-center justify-center flex-shrink-0 mt-0.5"
        style={{ background: 'var(--color-accent-soft)', color: 'var(--color-accent)', border: '1px solid color-mix(in srgb, var(--color-accent) 35%, transparent)' }}
      >
        <PageIcon size={18} strokeWidth={1.8} />
      </div>
      <div>
        <h1 className="font-display text-2xl text-ink mb-1 leading-tight">{title}</h1>
        {children && <p className="text-ink-dim text-sm max-w-3xl">{children}</p>}
      </div>
    </div>
  )
}
