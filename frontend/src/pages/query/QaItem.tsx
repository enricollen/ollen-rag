import { useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Button } from '../../components/Button'
import { ChunkText } from '../../components/ChunkText'
import { Pill } from '../../components/Pill'
import type { QaHistoryEntry } from '../../store/historyStore'

// Splits an answer on inline "[n]" citation markers and turns each into a clickable span that
// jumps to the matching source card below.
function renderAnswer(answer: string, onCite: (n: string) => void): ReactNode {
  const parts: ReactNode[] = []
  let last = 0
  const re = /\[(\d+)\]/g
  let m: RegExpExecArray | null
  let key = 0
  while ((m = re.exec(answer))) {
    if (m.index > last) parts.push(<span key={key++}>{answer.slice(last, m.index)}</span>)
    const n = m[1]
    parts.push(
      <span key={key++} className="text-signal font-bold cursor-pointer" onClick={() => onCite(n)}>
        [{n}]
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < answer.length) parts.push(<span key={key++}>{answer.slice(last)}</span>)
  return parts
}

export function QaItem({ entry }: { entry: QaHistoryEntry }) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  function scrollToSource(id: string | number) {
    setOpen(true)
    const el = containerRef.current?.querySelector<HTMLElement>(`[data-sid="${id}"]`)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      el.style.borderLeftColor = 'var(--color-signal)'
      setTimeout(() => {
        el.style.borderLeftColor = ''
      }, 1200)
    }
  }

  return (
    <div className="mb-6" ref={containerRef}>
      <div className="font-semibold text-ink mb-2">{entry.query}</div>
      <div className="bg-surface-2/60 border border-line rounded-panel p-4 text-[0.92rem] text-ink/90 whitespace-pre-wrap">
        {renderAnswer(entry.answer, (n) => scrollToSource(n))}
      </div>
      <div className="mt-2.5">
        <Button variant="ghost" onClick={() => setOpen((v) => !v)} className="!px-2 !py-1 text-xs">
          {open ? 'Hide' : 'Show'} {entry.sources?.length || 0} source(s)
        </Button>
      </div>
      {open && (
        <div className="mt-2.5">
          {(entry.sources || []).map((s) => (
            <div key={s.id} data-sid={s.id} className="border-l-[3px] border-accent pl-3 py-2 mb-2 bg-surface-2/40 rounded-r-panel transition-colors">
              <span className="font-bold text-signal text-xs mr-1.5">[{s.id}]</span>
              <ChunkText text={s.text} />
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {typeof s.metadata?.file_name === 'string' && <Pill tone="soft">{s.metadata.file_name}</Pill>}
                {typeof s.metadata?.bucket === 'string' && <Pill>bucket: {s.metadata.bucket}</Pill>}
                <Pill>score {Number(s.score ?? 0).toFixed(3)}</Pill>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
