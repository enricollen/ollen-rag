import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'

const CONTROL_CLS =
  'w-full bg-surface-2 border border-line rounded-control px-3 py-2 text-sm text-ink font-sans transition-colors focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 placeholder:text-ink-faint'

export function Field({
  label,
  hint,
  className = '',
  children,
}: {
  label?: ReactNode
  hint?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <label className={`block mb-3.5 ${className}`}>
      {label && <span className="block text-xs font-medium text-ink-dim mb-1.5">{label}</span>}
      {children}
      {hint && <div className="text-[0.7rem] text-ink-faint mt-1">{hint}</div>}
    </label>
  )
}

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} type={props.type ?? 'text'} className={`${CONTROL_CLS} ${props.className ?? ''}`} />
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${CONTROL_CLS} ${props.className ?? ''}`} />
}

export function Checkbox({ label, ...props }: InputHTMLAttributes<HTMLInputElement> & { label: ReactNode }) {
  return (
    <label className="flex items-center gap-2 text-sm text-ink cursor-pointer select-none">
      <input {...props} type="checkbox" className="w-4 h-4 accent-[var(--color-accent)]" />
      {label}
    </label>
  )
}
