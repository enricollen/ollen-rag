import type { ButtonHTMLAttributes } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
}

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-accent text-white hover:brightness-110 shadow-[0_0_0_1px_rgba(0,0,0,0.15),var(--glow-accent)] font-semibold',
  secondary: 'bg-surface-2 text-ink border border-line hover:border-accent/60 hover:text-ink',
  ghost: 'bg-transparent text-ink-dim hover:text-ink hover:bg-white/5',
  danger: 'bg-transparent text-bad border border-bad/50 hover:bg-bad/10',
}

export function Button({ variant = 'secondary', className = '', type = 'button', ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      type={type}
      className={`inline-flex items-center justify-center gap-2 rounded-control px-4 py-2 text-sm font-medium transition-all duration-150 active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed ${VARIANTS[variant]} ${className}`}
    />
  )
}
