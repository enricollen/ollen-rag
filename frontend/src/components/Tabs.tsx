import type { ReactNode } from 'react'

export interface TabDef<T extends string> {
  id: T
  label: ReactNode
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
  size = 'md',
}: {
  tabs: TabDef<T>[]
  active: T
  onChange: (id: T) => void
  size?: 'md' | 'sm'
}) {
  return (
    <div className={`flex gap-2 ${size === 'md' ? 'mb-5' : ''}`}>
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => onChange(t.id)}
          className={`flex-1 rounded-panel border font-semibold transition-colors ${
            size === 'md' ? 'px-4 py-2.5 text-sm' : 'px-3 py-1.5 text-xs flex-initial'
          } ${
            active === t.id
              ? 'border-accent text-accent bg-accent-soft'
              : 'border-line text-ink-dim hover:border-accent/50'
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}
