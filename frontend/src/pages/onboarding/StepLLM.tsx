import { motion } from 'motion/react'
import { Button } from '../../components/Button'
import { LLM_CHOICES } from './providers'
import { ProviderCard } from './ProviderCard'

export function StepLLM({
  selected,
  onSelect,
  onNext,
  computeNote,
}: {
  selected: string
  onSelect: (id: string) => void
  onNext: () => void
  computeNote: string
}) {
  return (
    <div>
      <motion.h1
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-display text-3xl text-ink mb-2"
      >
        Set up ollen-rag
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="text-ink-dim mb-8">
        Choose how answers get generated. Local needs no account.
      </motion.p>
      <div className="flex flex-col gap-3">
        {LLM_CHOICES.map((c, i) => (
          <motion.div
            key={c.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 + i * 0.05 }}
          >
            <ProviderCard label={c.label} description={c.description} selected={selected === c.id} onClick={() => onSelect(c.id)} />
          </motion.div>
        ))}
      </div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-8 flex items-center justify-between">
        {computeNote ? <span className="text-xs text-ink-faint">{computeNote}</span> : <span />}
        <Button variant="primary" onClick={onNext} className="px-6">
          Next
        </Button>
      </motion.div>
    </div>
  )
}
