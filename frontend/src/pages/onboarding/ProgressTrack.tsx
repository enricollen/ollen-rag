import { motion } from 'motion/react'

// Thin segmented glowing tracker across the top of the wizard stage -- replaces a plain step
// counter with something that reads as "calibrating" rather than "form page 2 of 4".
export function ProgressTrack({ steps, currentIndex }: { steps: string[]; currentIndex: number }) {
  return (
    <div className="flex gap-1.5 mb-10">
      {steps.map((label, i) => (
        <div key={label} className="flex-1">
          <div className="h-[3px] rounded-full bg-line overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: 'linear-gradient(90deg, var(--color-accent), var(--color-signal))' }}
              initial={false}
              animate={{ width: i < currentIndex ? '100%' : i === currentIndex ? '45%' : '0%' }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
          <div className={`mt-1.5 text-[0.68rem] uppercase tracking-wide ${i <= currentIndex ? 'text-ink-dim' : 'text-ink-faint'}`}>
            {label}
          </div>
        </div>
      ))}
    </div>
  )
}
