import { AnimatePresence, motion } from 'motion/react'
import { useToastStore } from '../store/toastStore'

const BORDER: Record<string, string> = {
  info: 'border-l-signal',
  error: 'border-l-bad',
  success: 'border-l-good',
}

export function ToastHost() {
  const toasts = useToastStore((s) => s.toasts)
  const dismiss = useToastStore((s) => s.dismiss)
  return (
    <div className="fixed top-5 right-6 flex flex-col gap-2.5 z-[100] w-[340px]">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 24 }}
            onClick={() => dismiss(t.id)}
            className={`glass-panel border-l-2 ${BORDER[t.kind]} px-3.5 py-2.5 text-sm cursor-pointer shadow-lg`}
          >
            {t.message}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
