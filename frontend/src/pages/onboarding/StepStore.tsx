import { motion } from 'motion/react'
import { Button } from '../../components/Button'
import { ProviderCard } from './ProviderCard'

export function StepStore({
  selected,
  onSelect,
  onBack,
  onFinish,
}: {
  selected: string
  onSelect: (id: string) => void
  onBack: () => void
  onFinish: () => void
}) {
  return (
    <div>
      <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="font-display text-3xl text-ink mb-2">
        Vector store
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="text-ink-dim mb-8">
        Where retrieved chunks get indexed.
      </motion.p>
      <div className="flex flex-col gap-3">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <ProviderCard
            label="Chroma"
            description="On-disk, nothing else to run — recommended for a first look."
            selected={selected === 'chroma'}
            onClick={() => onSelect('chroma')}
          />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 }}>
          <ProviderCard
            label="Qdrant"
            description="Needs the qdrant compose profile running."
            selected={selected === 'qdrant'}
            onClick={() => onSelect('qdrant')}
          />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.22 }}>
          <ProviderCard
            label="OpenSearch"
            description="Dense + BM25 hybrid — needs the opensearch compose profile."
            selected={selected === 'opensearch'}
            onClick={() => onSelect('opensearch')}
          />
        </motion.div>
      </div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }} className="flex items-center justify-between mt-8">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
        <Button variant="primary" onClick={onFinish} className="px-6">
          Finish
        </Button>
      </motion.div>
    </div>
  )
}
