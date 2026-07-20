import { useState } from 'react'
import { InfoIcon } from '../../components/icons'
import { METRIC_DEFS } from './metricDefs'

// Hidden-by-default glossary: a <details> the user clicks to reveal a grid of metric explanations.
export function MetricsLegend() {
  const [open, setOpen] = useState(false)
  return (
    <details open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)} className="mb-4 border border-line rounded-panel overflow-hidden">
      <summary className="px-4 py-3 cursor-pointer text-sm text-ink-dim flex items-center gap-2">
        <InfoIcon size={14} className="text-signal" /> What do these metrics mean?
        <span className="text-ink-faint text-xs">(click to show &mdash; or hover any metric label)</span>
      </summary>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 px-4 pb-4">
        {METRIC_DEFS.map((d) => (
          <div key={d.key} className="bg-surface-2/60 border border-line rounded-panel p-3">
            <div className="font-bold text-sm text-ink mb-1">{d.name}</div>
            <div className="text-sm text-ink-dim mb-1.5">{d.long}</div>
            <div className="text-xs text-ink-faint">
              <span className="text-signal font-semibold">e.g.</span> {d.example}
            </div>
          </div>
        ))}
      </div>
    </details>
  )
}
