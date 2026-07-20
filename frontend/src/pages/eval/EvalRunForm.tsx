import { useEffect, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { EvalLegReport, EvalReport, IndexInfo, IndexListEntry } from '../../api/types'
import { Button } from '../../components/Button'
import { Checkbox, Field, Select, TextInput } from '../../components/Field'
import { IndexInfoPanel } from '../../components/IndexInfoPanel'
import { Spinner } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { toast } from '../../store/toastStore'
import { EvalLegReportView } from './EvalLegReportView'
import { EvalReportView } from './EvalReportView'

function isLegReport(res: EvalReport | EvalLegReport): res is EvalLegReport {
  return 'per_leg' in res
}

// The eval config form (index/dataset/k/threshold/per-leg/save) plus its own result rendering --
// self-contained so EvalPage just composes this next to the run-history panel. Calls `onRunSaved`
// whenever the backend actually persisted a run, so the history list can refresh.
export function EvalRunForm({ onRunSaved }: { onRunSaved?: () => void }) {
  const [indices, setIndices] = useState<IndexListEntry[]>([])
  const [previewIndex, setPreviewIndex] = useState('')
  const [previewInfo, setPreviewInfo] = useState<IndexInfo | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [dataset, setDataset] = useState('example_bucket')
  const [k, setK] = useState('5')
  const [threshold, setThreshold] = useState('')
  const [perLeg, setPerLeg] = useState(false)
  const [save, setSave] = useState(false)
  const [label, setLabel] = useState('')
  const [running, setRunning] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [result, setResult] = useState<EvalReport | EvalLegReport | null>(null)
  const [errorBox, setErrorBox] = useState('')

  useEffect(() => {
    endpoints
      .indices()
      .then((r) => setIndices(r.indices))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!previewIndex) {
      setPreviewInfo(null)
      return
    }
    setPreviewLoading(true)
    endpoints
      .indexInfo(previewIndex)
      .then(setPreviewInfo)
      .catch((e) => {
        setPreviewInfo(null)
        toast(errorMessage(e), 'error')
      })
      .finally(() => setPreviewLoading(false))
  }, [previewIndex])

  async function submit() {
    if (!dataset.trim()) {
      toast('Enter a dataset name', 'error')
      return
    }
    const kNum = Number(k) || 5
    setRunning(true)
    setStatusText('running…')
    setResult(null)
    setErrorBox('')
    try {
      const body: Record<string, unknown> = { dataset: dataset.trim(), k: kNum, top_k: kNum }
      if (previewIndex) body.index_name = previewIndex
      if (threshold !== '') body.similarity_threshold = Number(threshold)
      if (perLeg) body.per_leg = true
      if (save && !perLeg) {
        body.save = true
        if (label.trim()) body.label = label.trim()
      }
      const res = await endpoints.evalRetrieval(body as never)
      setResult(res)
      if (isLegReport(res)) {
        setStatusText('per-leg done')
      } else {
        const overall = res.overall ?? ({} as never)
        const nCases = overall.cases ?? res.cases?.length ?? '?'
        const hitCount = res.cases?.filter((c) => c.matched > 0).length ?? '?'
        setStatusText(`${hitCount}/${nCases} hits`)
        if (res.run_id) {
          toast(`Saved run ${res.run_id}`, 'success')
          onRunSaved?.()
        }
      }
    } catch (e) {
      setStatusText('')
      setErrorBox(errorMessage(e))
      toast(errorMessage(e), 'error')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div>
      <Panel>
        <Field label="Index" hint="all cases run against this index, using its locked embedding model; leave empty to use each case's own index from the dataset">
          <Select value={previewIndex} onChange={(e) => setPreviewIndex(e.target.value)}>
            <option value="">(use dataset's per-case index)</option>
            {indices.map((ix) => (
              <option key={ix.index} value={ix.index}>
                {ix.index} ({ix['docs.count']} docs)
              </option>
            ))}
          </Select>
        </Field>
        {previewLoading && (
          <div className="text-sm text-ink-dim flex items-center gap-2 mb-3">
            <Spinner /> loading&hellip;
          </div>
        )}
        {previewInfo && <IndexInfoPanel info={previewInfo} verb="Eval cases" />}

        <Field
          label="Dataset name"
          hint={
            <>
              stem only, e.g. <code className="bg-surface-2 px-1 py-0.5 rounded">example_bucket</code> &rarr;{' '}
              <code className="bg-surface-2 px-1 py-0.5 rounded">config/eval/example_bucket.yaml</code>
            </>
          }
        >
          <TextInput value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="example_bucket" />
        </Field>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field label="k (recall@k)">
            <TextInput type="number" min={1} max={50} value={k} onChange={(e) => setK(e.target.value)} />
          </Field>
          <Field label="similarity_threshold" hint="optional override">
            <TextInput type="number" min={0} max={1} step={0.01} placeholder="from settings" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
          </Field>
        </div>

        <div className="flex items-center gap-6 flex-wrap mt-2 mb-4">
          <Checkbox
            label={<>Per-leg attribution <span className="text-ink-faint text-xs">(bm25/dense/hybrid/reranked + rerank lift)</span></>}
            checked={perLeg}
            onChange={(e) => {
              setPerLeg(e.target.checked)
              if (e.target.checked) setSave(false)
            }}
          />
          <Checkbox
            label={
              <>
                Save run{' '}
                {perLeg && <span className="text-ink-faint text-xs">(not available with per-leg attribution)</span>}
              </>
            }
            checked={save}
            disabled={perLeg}
            onChange={(e) => setSave(e.target.checked)}
          />
          <TextInput className="max-w-[16rem]" placeholder="run label (optional)" value={label} onChange={(e) => setLabel(e.target.value)} />
        </div>

        <div className="flex items-center gap-4">
          <Button variant="primary" onClick={submit} disabled={running}>
            {running ? <Spinner /> : null} Run eval
          </Button>
          {statusText && <span className="text-sm text-ink-dim">{statusText}</span>}
        </div>
      </Panel>

      {errorBox && (
        <Panel className="mt-4">
          <div className="text-bad text-sm">{errorBox}</div>
        </Panel>
      )}

      {result && (isLegReport(result) ? <EvalLegReportView res={result} /> : <EvalReportView res={result} />)}
    </div>
  )
}
