import { useEffect, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { CompareResponse, EvalRunSummary } from '../../api/types'
import { Button } from '../../components/Button'
import { Select } from '../../components/Field'
import { Spinner } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { toast } from '../../store/toastStore'
import { EvalCompareView } from './EvalCompareView'
import { RunMeta } from './RunMeta'
import { runOptionLabel } from './runFormat'

// Saved-run picker + paired A/B comparison. `runs` is owned by the parent page (so a newly saved
// run from `EvalRunForm` can trigger a refresh here too) -- this component only owns the A/B
// selection and the compare-result state.
export function EvalRunHistory({ runs, onRefresh }: { runs: EvalRunSummary[]; onRefresh: () => void }) {
  const [runAId, setRunAId] = useState('')
  const [runBId, setRunBId] = useState('')
  const [comparing, setComparing] = useState(false)
  const [cmp, setCmp] = useState<CompareResponse | null>(null)

  // Default A to the second-newest run and B to the newest, without clobbering a selection the
  // user already made once the list has loaded.
  useEffect(() => {
    if (!runs.length) return
    setRunAId((prev) => prev || runs[1]?.id || runs[0].id)
    setRunBId((prev) => prev || runs[0].id)
  }, [runs])

  const runsById = Object.fromEntries(runs.map((r) => [r.id, r]))

  async function runCompare() {
    if (!runAId || !runBId) {
      toast('Save at least two runs to compare', 'error')
      return
    }
    if (runAId === runBId) {
      toast('Pick two different runs', 'error')
      return
    }
    setComparing(true)
    setCmp(null)
    try {
      const res = await endpoints.evalCompare(runAId, runBId)
      setCmp(res)
    } catch (e) {
      toast(errorMessage(e), 'error')
    } finally {
      setComparing(false)
    }
  }

  return (
    <Panel
      title="Run history &amp; A/B compare"
      className="mt-6"
      subtitle={
        <>
          Every eval you save (tick <code className="bg-surface-2 px-1.5 py-0.5 rounded">Save run</code> above) is stored here. Pick two and hit
          compare to see whether a change &mdash; different embedding model, chunking, threshold &mdash; actually moved the numbers. Runs are
          matched <strong>per query</strong>, so the delta is a fair like-for-like, and each metric is flagged <em>significant</em> when its
          confidence interval clears zero (i.e. unlikely to be luck).
        </>
      }
    >
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-start">
        <div>
          <div className="text-xs text-ink-faint mb-1">
            A &middot; Baseline <span className="italic">what you measure against</span>
          </div>
          <Select value={runAId} onChange={(e) => setRunAId(e.target.value)}>
            {!runs.length && <option value="">no saved runs</option>}
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {runOptionLabel(r)}
              </option>
            ))}
          </Select>
          <div className="mt-1.5">
            <RunMeta run={runsById[runAId]} />
          </div>
        </div>
        <div className="hidden md:flex items-center justify-center text-ink-faint pt-6" aria-hidden>
          &rarr;
        </div>
        <div>
          <div className="text-xs text-ink-faint mb-1">
            B &middot; Variant <span className="italic">the change you're testing</span>
          </div>
          <Select value={runBId} onChange={(e) => setRunBId(e.target.value)}>
            {!runs.length && <option value="">no saved runs</option>}
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {runOptionLabel(r)}
              </option>
            ))}
          </Select>
          <div className="mt-1.5">
            <RunMeta run={runsById[runBId]} />
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3 mt-4">
        <Button variant="primary" onClick={runCompare} disabled={comparing || !runs.length}>
          {comparing ? <Spinner /> : null} Compare A &rarr; B
        </Button>
        <Button variant="secondary" onClick={onRefresh}>
          Refresh list
        </Button>
        {!runs.length && (
          <span className="text-sm text-ink-faint">No saved runs yet &mdash; tick &ldquo;Save run&rdquo; above, run an eval, then come back.</span>
        )}
      </div>
      {cmp && (
        <EvalCompareView
          cmp={cmp}
          runA={runsById[runAId]}
          runB={runsById[runBId]}
          labelA={runsById[runAId] ? runOptionLabel(runsById[runAId]) : 'A'}
          labelB={runsById[runBId] ? runOptionLabel(runsById[runBId]) : 'B'}
        />
      )}
    </Panel>
  )
}
