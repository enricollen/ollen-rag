import type { SettingsDump } from '../../api/types'
import { Pill } from '../../components/Pill'

// key -> " (reuses X)" note for providers that just delegate to another section's credentials.
const REUSE_NOTE: Record<string, string> = { 'litellm-watsonx': ' (reuses watsonx §2)', 'litellm-ollama': ' (reuses ollama §4)' }

export function WiringBanner({ current }: { current: SettingsDump }) {
  const llm = String(current.llm_provider ?? '')
  const emb = String(current.embedding_provider ?? '')
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      <Pill tone="ok">
        LLM: {llm}
        {REUSE_NOTE[llm] ?? ''}
      </Pill>
      <Pill tone="ok">
        Embeddings: {emb}
        {REUSE_NOTE[emb] ?? ''}
      </Pill>
      <Pill tone="ok">Reranker: {String(current.reranker_provider ?? '')}</Pill>
      <Pill tone="ok">Vectors: {String(current.vector_store ?? '')}</Pill>
    </div>
  )
}
