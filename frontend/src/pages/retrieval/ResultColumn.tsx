import type { RetrievedNode } from '../../api/types'
import { EmptyState } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { NodeCard } from './NodeCard'

export function ResultColumn({ title, subtitle, nodes }: { title: string; subtitle: string; nodes: RetrievedNode[] }) {
  return (
    <Panel title={<>{title} <span className="text-ink-faint font-normal text-xs">({nodes.length})</span></>} subtitle={subtitle}>
      {nodes.length ? nodes.map((n, i) => <NodeCard key={i} node={n} />) : <EmptyState>No matching nodes.</EmptyState>}
    </Panel>
  )
}
