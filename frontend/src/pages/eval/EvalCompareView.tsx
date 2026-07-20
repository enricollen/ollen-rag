import type { CompareResponse, EvalRunSummary } from '../../api/types'
import { FolderOpenIcon } from '../../components/icons'
import { Panel } from '../../components/Panel'
import { Pill } from '../../components/Pill'
import { chunkingSummary } from '../../lib/format'
import { DeltaChart } from './charts'

function SystemRow({ run, letter }: { run: EvalRunSummary | undefined; letter: string }) {
  const sys = run?.params?.system
  if (!sys) {
    return (
      <div className="flex gap-2 text-sm py-1">
        <span className="font-bold text-ink-dim w-4">{letter}</span>
        <span className="text-ink-faint italic">system config not recorded (older run)</span>
      </div>
    )
  }
  const indices = Object.entries(sys.indices ?? {})
  return (
    <div className="mb-1.5">
      <div className="flex gap-2 text-sm py-1">
        <span className="font-bold text-ink-dim w-4">{letter}</span>
        <span className="text-ink flex items-center gap-1.5">
          <FolderOpenIcon size={13} className="text-signal" /> {sys.vector_store ?? '?'}
        </span>
      </div>
      {indices.map(([idx, m]) => (
        <div key={idx} className="text-xs text-ink-faint ml-6">
          <code className="bg-surface-2 px-1 py-0.5 rounded">{idx}</code> &mdash;{' '}
          {m.embedding_provider ? `${m.embedding_provider}/${m.embedding_model}` : 'unrecorded'} &middot; {chunkingSummary(m.chunking)}
        </div>
      ))}
    </div>
  )
}

export function EvalCompareView({
  cmp,
  runA,
  runB,
  labelA,
  labelB,
}: {
  cmp: CompareResponse
  runA?: EvalRunSummary
  runB?: EvalRunSummary
  labelA: string
  labelB: string
}) {
  return (
    <div className="mt-4 flex flex-col gap-4">
      <Panel title="System & params" subtitle="What each run actually hit &mdash; so a metric delta is legible as one system vs another, not just numbers.">
        <SystemRow run={runA} letter="A" />
        <SystemRow run={runB} letter="B" />
      </Panel>
      <Panel
        title={<>A/B comparison <span className="text-ink-faint text-sm font-normal">({cmp.n_paired} paired cases)</span></>}
        subtitle={
          <>
            Delta = <strong>{labelB}</strong> &minus; <strong>{labelA}</strong>, paired by query. 95% bootstrap CI; a metric is{' '}
            <em>significant</em> when its CI excludes 0.
          </>
        }
      >
        <table className="w-full text-sm border-collapse mb-3">
          <thead>
            <tr>
              {['Metric', 'Δ (B−A)', '95% CI', ''].map((h) => (
                <th key={h} className="text-left py-1.5 px-2 text-ink-dim font-semibold border-b border-line">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(cmp.metrics ?? {}).map(([m, d]) => {
              const cls = d.delta > 0 ? 'text-good' : d.delta < 0 ? 'text-bad' : 'text-ink-dim'
              return (
                <tr key={m}>
                  <td className="py-1.5 px-2 border-b border-line text-ink-dim">{m}</td>
                  <td className={`py-1.5 px-2 border-b border-line font-mono ${cls}`}>
                    {d.delta >= 0 ? '+' : ''}
                    {d.delta.toFixed(4)}
                  </td>
                  <td className="py-1.5 px-2 border-b border-line font-mono">
                    [{d.ci[0].toFixed(3)}, {d.ci[1].toFixed(3)}]
                  </td>
                  <td className="py-1.5 px-2 border-b border-line">
                    {d.significant ? <Pill tone="ok">significant</Pill> : <span className="text-ink-faint text-xs">n.s.</span>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {(cmp.a_only?.length || cmp.b_only?.length) ? (
          <div className="text-xs text-ink-faint mb-3">
            unpaired: {cmp.a_only?.length ?? 0} only in A, {cmp.b_only?.length ?? 0} only in B (excluded)
          </div>
        ) : null}
        <DeltaChart cmp={cmp} />
      </Panel>
    </div>
  )
}
