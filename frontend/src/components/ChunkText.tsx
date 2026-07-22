import { useLayoutEffect, useRef, useState } from 'react'

// Renders a retrieved chunk's full text inside a visually-clamped box (never truncated in the
// DOM), revealing an Expand/Collapse toggle only when the text actually overflows the clamp.
export function ChunkText({ text, className = '' }: { text: string; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [overflowing, setOverflowing] = useState(false)
  const [expanded, setExpanded] = useState(false)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setOverflowing(el.scrollHeight - el.clientHeight > 2)
  }, [text])

  return (
    <div className={className}>
      <div
        ref={ref}
        className={`whitespace-pre-wrap text-sm text-ink/90 ${!expanded && overflowing ? 'max-h-[9em] overflow-hidden' : ''}`}
        style={!expanded && overflowing ? { maskImage: 'linear-gradient(180deg, #000 68%, transparent)' } : undefined}
      >
        {text}
      </div>
      {overflowing && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1.5 text-[0.76rem] font-semibold text-signal hover:underline"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      )}
    </div>
  )
}
