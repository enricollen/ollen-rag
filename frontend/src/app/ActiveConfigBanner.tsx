import { useEffect, useState } from 'react'
import { endpoints } from '../api/client'
import type { ActiveComponentSummary } from '../api/types'
import { BarChartIcon, CpuIcon, DatabaseIcon, GearIcon, MessageSquareIcon, ScissorsIcon, TargetIcon, type IconProps } from '../components/icons'

const ITEMS: { icon: (props: IconProps) => React.ReactElement; label: string; value: (a: ActiveComponentSummary) => string }[] = [
  { icon: MessageSquareIcon, label: 'LLM', value: (a) => `${a.llm.provider} · ${a.llm.model}` },
  { icon: CpuIcon, label: 'Embedding', value: (a) => `${a.embedding.provider} · ${a.embedding.model}` },
  { icon: TargetIcon, label: 'Reranker', value: (a) => `${a.reranker.provider} · ${a.reranker.model}` },
  { icon: DatabaseIcon, label: 'Vector store', value: (a) => a.vector_store },
  { icon: ScissorsIcon, label: 'Chunking', value: (a) => `${a.chunking.strategy} · ${a.chunking.chunk_size}/${a.chunking.chunk_overlap}` },
  { icon: BarChartIcon, label: 'top_k / top_n', value: (a) => `${a.retrieval_top_k} / ${a.rerank_top_n}` },
]

// Always-on readout of the live wiring the whole console runs on, sourced from /api/v1/config's
// `active` block (same resolution as the startup log). Purely informational. Refetches whenever
// `refreshKey` changes (the Shell bumps it on every route change).
export function ActiveConfigBanner({ refreshKey }: { refreshKey: unknown }) {
  const [active, setActive] = useState<ActiveComponentSummary | null>(null)

  useEffect(() => {
    let cancelled = false
    endpoints
      .config()
      .then((cfg) => {
        if (!cancelled) setActive(cfg.active)
      })
      .catch(() => {
        if (!cancelled) setActive(null)
      })
    return () => {
      cancelled = true
    }
  }, [refreshKey])

  if (!active) return null
  return (
    <div className="glass-panel px-3.5 py-2.5 mb-5" title="Active configuration — change it in Settings">
      <div className="text-xs font-bold tracking-wide text-ink mb-2 flex items-center gap-1.5">
        <GearIcon size={13} className="text-accent" /> Active Configuration
      </div>
      <div className="flex flex-wrap gap-2 items-center">
        {ITEMS.map((item) => {
          const ItemIcon = item.icon
          return (
            <span key={item.label} className="inline-flex items-center gap-1.5 text-[0.72rem] px-2 py-0.5 rounded-full bg-surface-2 border border-line">
              <ItemIcon size={12} className="text-signal" />
              <span className="text-ink-faint uppercase tracking-wide text-[0.64rem] font-semibold">{item.label}</span>
              <span className="text-ink font-medium">{item.value(active)}</span>
            </span>
          )
        })}
      </div>
    </div>
  )
}
