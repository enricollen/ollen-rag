import { motion } from 'motion/react'
import type { ReactNode } from 'react'

export function ProviderCard({
  label,
  description,
  selected,
  onClick,
}: {
  label: string
  description: string
  selected: boolean
  onClick: () => void
}) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileHover={{ y: -3 }}
      whileTap={{ scale: 0.98 }}
      className={`relative text-left w-full rounded-2xl border px-5 py-4 transition-colors ${
        selected ? 'border-accent bg-accent-soft' : 'border-line bg-surface-2/40 hover:border-accent/50'
      }`}
      style={selected ? { boxShadow: 'var(--glow-accent)' } : undefined}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="font-display text-[0.95rem] text-ink">{label}</span>
        <CheckBadge show={selected} />
      </div>
      <div className="text-sm text-ink-dim mt-1">{description}</div>
    </motion.button>
  )
}

function CheckBadge({ show }: { show: boolean }) {
  return (
    <span
      className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 border transition-colors"
      style={{
        borderColor: show ? 'var(--color-accent)' : 'var(--color-line-strong)',
        background: show ? 'var(--color-accent)' : 'transparent',
      }}
    >
      {show && <Check />}
    </span>
  )
}

function Check(): ReactNode {
  return (
    <motion.svg
      viewBox="0 0 16 16"
      width="11"
      height="11"
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 500, damping: 25 }}
    >
      <path d="M2 8.5 6 12.5 14 3.5" stroke="#fff" strokeWidth="2.2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </motion.svg>
  )
}
