import { motion } from 'motion/react'
import { Spinner } from '../../components/Misc'
import { XIcon } from '../../components/icons'

export type FinishState = 'saving' | 'restarting' | 'done' | 'error'

export function StepFinish({ state, detail }: { state: FinishState; detail?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {(state === 'saving' || state === 'restarting') && (
        <>
          <Spinner className="w-8 h-8 border-[3px] mb-5" />
          <div className="font-display text-xl text-ink">{state === 'saving' ? 'Saving configuration…' : 'Restarting service…'}</div>
          <div className="text-ink-dim text-sm mt-2">This usually takes a few seconds.</div>
        </>
      )}
      {state === 'done' && (
        <>
          <SuccessGlyph />
          <div className="font-display text-xl text-ink mt-4">All set</div>
          <div className="text-ink-dim text-sm mt-2 max-w-sm">
            {'Configuration applied — no restart needed. Taking you to the console…'}
          </div>
        </>
      )}
      {state === 'error' && (
        <>
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center text-white"
            style={{ background: 'var(--color-bad)', boxShadow: '0 0 40px -4px color-mix(in srgb, var(--color-bad) 60%, transparent)' }}
          >
            <XIcon size={26} strokeWidth={2.6} />
          </div>
          <div className="font-display text-xl text-ink mt-4">Save failed</div>
          <div className="text-bad text-sm mt-2 max-w-sm">{detail}</div>
        </>
      )}
    </div>
  )
}

function SuccessGlyph() {
  return (
    <motion.div
      initial={{ scale: 0.5, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className="w-16 h-16 rounded-full flex items-center justify-center"
      style={{ background: 'var(--color-good)', boxShadow: '0 0 40px -4px color-mix(in srgb, var(--color-good) 60%, transparent)' }}
    >
      <svg viewBox="0 0 24 24" width="28" height="28">
        <path d="M4 12.5 9.5 18 20 6" stroke="#0a0b0d" strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </motion.div>
  )
}
