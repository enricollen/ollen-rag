import { CHUNK_FIELDS, STRATEGIES } from '../lib/strategies'
import { AlertTriangleIcon } from './icons'
import { Pill } from './Pill'

export function StrategyPicker({ selected, onSelect }: { selected: string; onSelect: (name: string) => void }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {STRATEGIES.map((s) => (
        <div
          key={s.name}
          onClick={() => onSelect(s.name)}
          className={`bg-surface-2/50 border-[1.5px] rounded-panel p-3.5 cursor-pointer transition-colors ${
            selected === s.name ? 'border-accent bg-accent-soft' : 'border-line hover:border-line-strong'
          }`}
        >
          <div className="font-bold text-sm mb-1 capitalize flex items-center gap-1.5">
            {s.name}
            {s.warn && <Pill tone="warn">slow</Pill>}
          </div>
          <div className="text-xs text-ink-dim mb-2">{s.desc}</div>
          {s.warn && (
            <div className="text-[0.72rem] text-warn mb-1.5 flex items-center gap-1">
              <AlertTriangleIcon size={11} /> {s.warn}
            </div>
          )}
          <div className="text-[0.72rem] font-mono bg-surface px-2 py-1.5 rounded text-ink-faint whitespace-pre-wrap">{s.example}</div>
        </div>
      ))}
    </div>
  )
}

export function ChunkParamInputs({
  strategy,
  values,
  onChange,
}: {
  strategy: string
  values: Record<string, number | undefined>
  onChange: (values: Record<string, number | undefined>) => void
}) {
  const fields = CHUNK_FIELDS[strategy] || []
  if (!fields.length) return null
  return (
    <div className="flex gap-4 mt-3 flex-wrap">
      {fields.map((f) => (
        <label key={f.key} className="flex-1 min-w-[10rem]">
          <span className="block text-xs text-ink-dim mb-1.5">{f.label}</span>
          <input
            type="number"
            min={f.min ?? 0}
            max={f.max}
            step={f.step ?? 1}
            value={values[f.key] ?? ''}
            onChange={(e) => onChange({ ...values, [f.key]: e.target.value === '' ? undefined : Number(e.target.value) })}
            className="w-full bg-surface-2 border border-line rounded-control px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
          />
        </label>
      ))}
    </div>
  )
}
