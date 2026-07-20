import { useState } from 'react'
import type { EvalCase } from '../../api/types'
import { ChunkText } from '../../components/ChunkText'
import { CheckIcon, ChevronRightIcon, XIcon } from '../../components/icons'
import { Pill } from '../../components/Pill'
import { METRIC_TITLE } from './metricDefs'

function Metric({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div title={title} className="flex-1 flex flex-col items-center py-2 px-1 border-r border-line last:border-0">
      <span className="text-[0.68rem] uppercase tracking-wide text-ink-dim">{label}</span>
      <span className="text-[0.95rem] font-bold text-ink mt-0.5">{value}</span>
    </div>
  )
}

export function EvalCaseCard({ c, forceOpen }: { c: EvalCase; forceOpen?: boolean }) {
  const [open, setOpen] = useState(false)
  const isOpen = forceOpen ?? open
  const hit = c.matched > 0

  return (
    <div className={`border rounded-panel mb-3 overflow-hidden border-line border-l-[3px] ${hit ? 'border-l-good' : 'border-l-bad'}`}>
      <div className="flex justify-between items-start gap-4 px-4 pt-3.5 pb-2.5">
        <div className="text-[0.95rem] font-semibold text-ink flex-1">{c.query}</div>
        <div className="flex items-center gap-2.5 flex-shrink-0">
          {c.bucket && <Pill tone="soft">bucket: {c.bucket}</Pill>}
          {hit ? (
            <Pill tone="ok">
              <CheckIcon size={11} /> HIT rank {c.first_rank}
            </Pill>
          ) : (
            <Pill tone="bad">
              <XIcon size={11} /> MISS
            </Pill>
          )}
        </div>
      </div>
      <div className="flex border-t border-b border-line">
        <Metric label="Recall" value={`${(c.recall * 100).toFixed(0)}%`} title={METRIC_TITLE.recall} />
        <Metric label="P@5" value={`${((c.precision_at?.['5'] ?? 0) * 100).toFixed(0)}%`} title={METRIC_TITLE.precision} />
        <Metric
          label="RR"
          value={<span className="font-mono">{(c.reciprocal_rank ?? 0).toFixed(3)}</span>}
          title="Reciprocal rank — 1 / rank of this query's first correct hit (rank 2 → 0.5)."
        />
        <Metric label="nDCG" value={<span className="font-mono">{(c.ndcg ?? 0).toFixed(3)}</span>} title={METRIC_TITLE.ndcg} />
        <Metric
          label="AP"
          value={<span className="font-mono">{(c.average_precision ?? 0).toFixed(3)}</span>}
          title="Average precision — this query's contribution to MAP (precision averaged at each relevant hit)."
        />
        <Metric label="Latency" value={<span className="font-mono">{(c.latency_ms ?? 0).toFixed(0)}ms</span>} title={METRIC_TITLE.latency} />
        <Metric
          label="Matched"
          value={`${c.matched}/${c.expected}`}
          title="Relevant sources found for this query vs how many were expected."
        />
      </div>
      <button type="button" onClick={() => setOpen((v) => !v)} className="w-full text-left px-4 py-2.5 text-sm text-ink-dim flex items-center gap-2">
        <ChevronRightIcon size={13} className={`transition-transform ${isOpen ? 'rotate-90' : ''}`} />
        Expected sources &amp; retrieved nodes
      </button>
      {isOpen && (
        <div className="px-4 pb-4 pt-1 border-t border-line">
          <div className="text-[0.7rem] uppercase tracking-wide text-ink-dim font-bold mb-1.5">
            Expected ({c.expected_chunks?.length ?? c.expected})
          </div>
          <div className="mb-3">
            {c.expected_chunks?.length ? (
              c.expected_chunks.map((e, i) => (
                <div key={i} className="flex items-center gap-2 mb-1.5 flex-wrap">
                  <Pill tone="soft">{e.file_name}</Pill>
                  {e.contains && <span className="text-sm text-ink-dim italic">"{e.contains}"</span>}
                </div>
              ))
            ) : (
              <em className="text-ink-faint">&mdash;</em>
            )}
          </div>
          <div className="text-[0.7rem] uppercase tracking-wide text-ink-dim font-bold mb-1.5">
            Retrieved nodes ({c.retrieved_nodes?.length ?? c.retrieved})
          </div>
          <div>
            {c.retrieved_nodes?.length ? (
              c.retrieved_nodes.map((n, i) => (
                <div key={i} className={`bg-surface-2/60 border border-line rounded-panel p-3 mb-2 ${n.matched ? 'border-good bg-good/5' : ''}`}>
                  <div className="flex items-center gap-2 flex-wrap mb-1.5">
                    <span className="text-[0.72rem] font-extrabold text-ink-dim min-w-[1.6rem]">#{n.rank}</span>
                    <Pill tone={n.matched ? 'ok' : 'default'}>{n.file_name}</Pill>
                    {n.score != null && <span className="font-mono text-xs text-ink-dim">score {n.score}</span>}
                    {n.matched && <Pill tone="ok">match</Pill>}
                  </div>
                  <ChunkText text={n.text} />
                </div>
              ))
            ) : (
              <em className="text-ink-faint">No nodes returned</em>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
