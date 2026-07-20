import type { RetrievedNode } from '../../api/types'
import { ChunkText } from '../../components/ChunkText'
import { Pill } from '../../components/Pill'
import { ScoreBar } from '../../components/ScoreBar'

export function NodeCard({ node }: { node: RetrievedNode & { retrieval_score?: number | null } }) {
  const metaChips = Object.entries(node.metadata || {}).filter(([k]) => k !== 'bucket')
  const bucket = node.metadata?.bucket as string | undefined
  return (
    <div className="bg-surface-2/60 border border-line rounded-panel p-3.5 mb-3">
      <div className="flex justify-between items-center mb-2 gap-3">
        <ScoreBar score={node.score ?? 0} />
      </div>
      <ChunkText text={node.text} />
      <div className="flex flex-wrap gap-1.5 mt-2">
        {bucket && <Pill tone="soft">bucket: {bucket}</Pill>}
        {node.score != null && <Pill title="cross-encoder rerank score">rerank {Number(node.score).toFixed(3)}</Pill>}
        {node.retrieval_score != null && <Pill title="fused hybrid score (pre-rerank)">hybrid {Number(node.retrieval_score).toFixed(3)}</Pill>}
        {metaChips.map(([k, v]) => (
          <Pill key={k}>
            {k}: {String(v)}
          </Pill>
        ))}
      </div>
    </div>
  )
}
