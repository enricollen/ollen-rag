import type { EvalSystem } from '../../api/types'

// Compact "which system" tag for a saved run: vector store + each resolved index's embedding
// model (index name only when it differs from a dataset run over a single index). Runs saved
// before this field existed have no params.system -- falls back silently.
export function systemTagText(system?: EvalSystem | null): string {
  if (!system) return ''
  const indices = Object.entries(system.indices ?? {})
  if (!indices.length) return system.vector_store ? ` · ${system.vector_store}` : ''
  const bits = indices.map(([idx, m]) => {
    const emb = m.embedding_provider ? `${m.embedding_provider}/${m.embedding_model}` : '?'
    return indices.length > 1 ? `${idx}:${emb}` : emb
  })
  return ` · ${system.vector_store}/${bits.join('+')}`
}
