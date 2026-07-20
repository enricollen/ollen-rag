import type { EvalRunSummary } from '../../api/types'

// Compact "which system" tag for a saved run: vector store + each resolved index's embedding
// model (index name only when it differs from a dataset run over a single index). Runs saved
// before this field existed have no params.system -- falls back silently.
export function systemTagShort(run: EvalRunSummary): string {
  const sys = run.params?.system
  if (!sys) return ''
  const indices = Object.entries(sys.indices ?? {})
  if (!indices.length) return sys.vector_store ? ` · ${sys.vector_store}` : ''
  const bits = indices.map(([idx, m]) => {
    const emb = m.embedding_provider ? `${m.embedding_provider}/${m.embedding_model}` : '?'
    return indices.length > 1 ? `${idx}:${emb}` : emb
  })
  return ` · ${sys.vector_store}/${bits.join('+')}`
}

// One <option> label for the run picker: timestamp + label + headline nDCG + system tag.
export function runOptionLabel(run: EvalRunSummary): string {
  const nd = run.overall?.ndcg != null ? ` · nDCG ${run.overall.ndcg.toFixed(3)}` : ''
  const lbl = run.label ? ` · ${run.label}` : ''
  return `${run.timestamp ?? run.id}${lbl}${nd}${systemTagShort(run)}`
}
