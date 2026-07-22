import { useEffect, useMemo, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { IndexListEntry, IndexVectorPoint, IndexVectorsResponse } from '../../api/types'
import { IndexSelect } from '../../components/IndexSelect'
import { BarChartIcon } from '../../components/icons'
import { EmptyState, Spinner } from '../../components/Misc'
import { Panel } from '../../components/Panel'
import { CHART_PALETTE } from '../../lib/format'
import { toast } from '../../store/toastStore'

const NO_VALUE_COLOR = '#898781'
const OTHER_COLOR = '#5c6479'

type ColorMode = 'bucket' | 'document'
const MODE_FIELD: Record<ColorMode, keyof IndexVectorPoint> = { bucket: 'bucket', document: 'file_name' }
const MODE_LABEL: Record<ColorMode, string> = { bucket: 'Bucket', document: 'Document' }
const MODE_NO_VALUE: Record<ColorMode, string> = { bucket: '(no bucket)', document: '(no document)' }

function buildColorMap(keys: string[]): Map<string, string> {
  const map = new Map<string, string>()
  keys.forEach((k, i) => map.set(k, i < CHART_PALETTE.length ? CHART_PALETTE[i] : OTHER_COLOR))
  return map
}

function fitToViewBox(points: IndexVectorPoint[], width: number, height: number, margin: number) {
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1
  const scaleX = (width - 2 * margin) / spanX
  const scaleY = (height - 2 * margin) / spanY
  return points.map((p) => ({ ...p, cx: margin + (p.x - minX) * scaleX, cy: margin + (p.y - minY) * scaleY }))
}

function gridLines(width: number, height: number, margin: number, cols = 6, rows = 4) {
  const color = 'var(--color-line)'
  const lines: React.ReactNode[] = []
  for (let i = 0; i <= cols; i++) {
    const x = margin + (i * (width - 2 * margin)) / cols
    lines.push(<line key={`c${i}`} x1={x} y1={margin} x2={x} y2={height - margin} stroke={color} strokeWidth={1} />)
  }
  for (let i = 0; i <= rows; i++) {
    const y = margin + (i * (height - 2 * margin)) / rows
    lines.push(<line key={`r${i}`} x1={margin} y1={y} x2={width - margin} y2={y} stroke={color} strokeWidth={1} />)
  }
  return lines
}

export function VisualizerTab() {
  const [indices, setIndices] = useState<IndexListEntry[]>([])
  const [selected, setSelected] = useState('')
  const [data, setData] = useState<IndexVectorsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [colorBy, setColorBy] = useState<ColorMode>('bucket')
  const [tooltip, setTooltip] = useState<{ x: number; y: number; point: IndexVectorPoint } | null>(null)

  useEffect(() => {
    endpoints
      .indices()
      .then((r) => setIndices(r.indices))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!selected) {
      setData(null)
      return
    }
    setLoading(true)
    endpoints
      .indexVectors(selected)
      .then(setData)
      .catch((e) => toast(errorMessage(e), 'error'))
      .finally(() => setLoading(false))
  }, [selected])

  const width = 720
  const height = 480
  const margin = 24
  const field = MODE_FIELD[colorBy]

  const keys = useMemo(() => {
    if (!data) return []
    if (colorBy === 'bucket') return data.buckets
    return [...new Set(data.points.map((p) => p.file_name).filter(Boolean) as string[])].sort()
  }, [data, colorBy])

  const colorMap = useMemo(() => buildColorMap(keys), [keys])
  const placed = useMemo(() => (data && data.points.length ? fitToViewBox(data.points, width, height, margin) : []), [data])

  function colorFor(point: IndexVectorPoint): string {
    const key = point[field] as string | null
    if (!key) return NO_VALUE_COLOR
    return colorMap.get(key) || OTHER_COLOR
  }

  const shownKeys = keys.slice(0, CHART_PALETTE.length)
  const overflow = keys.length > CHART_PALETTE.length

  return (
    <Panel
      title={
        <>
          <BarChartIcon size={14} className="text-signal inline -mt-0.5 mr-1" /> Visualizer
        </>
      }
      subtitle="2D projection (PCA) of an index's chunk embeddings, colored by bucket or document. Hover a point for its text."
    >
      <label className="block mb-4">
        <span className="block text-xs font-medium text-ink-dim mb-1.5">Index</span>
        <IndexSelect indices={indices} value={selected} onChange={setSelected} />
      </label>

      {!selected ? null : loading ? (
        <EmptyState>
          <Spinner />
        </EmptyState>
      ) : !data || data.points.length < 2 ? (
        <EmptyState>Not enough chunks to visualize yet (need at least 2).</EmptyState>
      ) : (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-ink-faint">Color by</span>
            <div className="inline-flex gap-1.5">
              {(['bucket', 'document'] as ColorMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setColorBy(m)}
                  className={`text-xs px-3 py-1.5 rounded-control border ${
                    colorBy === m ? 'border-accent text-accent bg-accent-soft' : 'border-line text-ink-dim'
                  }`}
                >
                  {MODE_LABEL[m]}
                </button>
              ))}
            </div>
          </div>
          <div className="text-xs text-ink-faint mb-2">{data.capped ? `showing ${data.returned} of ${data.total} chunks` : `${data.returned} chunk(s)`}</div>
          <div className="relative">
            <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width }} className="border border-line rounded-control bg-surface-2/60">
              {gridLines(width, height, margin)}
              {placed.map((p) => (
                <circle
                  key={p.id}
                  cx={p.cx}
                  cy={p.cy}
                  r={4}
                  fill={colorFor(p)}
                  fillOpacity={0.85}
                  onMouseEnter={(e) => setTooltip({ x: e.clientX, y: e.clientY, point: p })}
                  onMouseMove={(e) => setTooltip({ x: e.clientX, y: e.clientY, point: p })}
                  onMouseLeave={() => setTooltip(null)}
                  className="cursor-pointer"
                />
              ))}
            </svg>
            {tooltip && (
              <div
                className="fixed pointer-events-none bg-surface border border-line rounded-control px-3 py-2 max-w-[280px] text-sm z-10"
                style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
              >
                <div className="font-semibold text-ink">{tooltip.point.bucket || '(no bucket)'}</div>
                {tooltip.point.file_name && <div className="text-ink-dim">{tooltip.point.file_name}</div>}
                <div className="text-ink-faint text-xs">{tooltip.point.length} chars</div>
                <div className="mt-1 text-ink-dim">
                  {tooltip.point.text}
                  {tooltip.point.length > tooltip.point.text.length ? '…' : ''}
                </div>
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-3 mt-3">
            {shownKeys.map((k) => (
              <span key={k} className="inline-flex items-center gap-1.5 text-xs text-ink-dim">
                <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: colorMap.get(k) }} />
                {k}
              </span>
            ))}
            {overflow && (
              <span className="inline-flex items-center gap-1.5 text-xs text-ink-dim">
                <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: OTHER_COLOR }} />
                Other ({keys.length - shownKeys.length} more {colorBy === 'bucket' ? 'buckets' : 'documents'})
              </span>
            )}
            <span className="inline-flex items-center gap-1.5 text-xs text-ink-dim">
              <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: NO_VALUE_COLOR }} />
              {MODE_NO_VALUE[colorBy]}
            </span>
          </div>
        </div>
      )}
    </Panel>
  )
}
