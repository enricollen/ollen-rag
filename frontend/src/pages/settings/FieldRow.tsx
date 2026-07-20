import type { SettingsDump } from '../../api/types'
import { Select, TextInput } from '../../components/Field'
import { AlertTriangleIcon } from '../../components/icons'
import { Pill } from '../../components/Pill'
import { FIELD_WARN, selectionOf, type FieldDef } from './schema'

type Current = SettingsDump

function inputValueType(initial: Current, key: string): 'number' | 'boolean' | 'string' {
  const v = initial[key]
  if (typeof v === 'number') return 'number'
  if (typeof v === 'boolean') return 'boolean'
  return 'string'
}

export function FieldRow({
  field,
  current,
  initial,
  sectionActive,
  onChange,
}: {
  field: FieldDef
  current: Current
  initial: Current
  sectionActive: boolean
  onChange: (key: string, value: string | number | boolean) => void
}) {
  const val = current[field.key]
  const isEmpty = val === '' || val == null
  const needsValue = field.req && sectionActive && isEmpty
  const fieldInert = field.activeWhen ? !field.activeWhen(selectionOf(current)) : false
  const warn = FIELD_WARN[field.key]

  function readInput(raw: string | boolean) {
    const kind = inputValueType(initial, field.key)
    if (kind === 'number') return Number(raw)
    if (kind === 'boolean') return raw === true || raw === 'true'
    return raw
  }

  return (
    <div className={`flex justify-between gap-4 text-sm py-1.5 border-b border-dashed border-line last:border-0 ${fieldInert ? 'opacity-50' : ''}`}>
      <span className="text-ink-dim flex items-center gap-1.5 flex-shrink-0 pt-2">
        {field.key}
        {needsValue && <Pill tone="warn">needs value</Pill>}
        {fieldInert && <Pill>inactive</Pill>}
      </span>
      <span className="flex-1 max-w-[60%]">
        {field.type === 'select' ? (
          <Select value={String(val ?? '')} onChange={(e) => onChange(field.key, readInput(e.target.value))}>
            {(field.pick ?? []).map((o) => (
              <option key={o}>{o}</option>
            ))}
          </Select>
        ) : field.type === 'bool' ? (
          <Select value={val ? 'true' : 'false'} onChange={(e) => onChange(field.key, readInput(e.target.value === 'true'))}>
            <option value="true">true</option>
            <option value="false">false</option>
          </Select>
        ) : (
          <TextInput
            type={field.type === 'password' ? 'password' : field.type === 'number' ? 'number' : 'text'}
            step={field.type === 'number' ? 'any' : undefined}
            value={val == null ? '' : String(val)}
            onChange={(e) => onChange(field.key, readInput(e.target.value))}
          />
        )}
        {warn && (
          <div className="text-[0.74rem] text-warn mt-1 flex items-center gap-1">
            <AlertTriangleIcon size={11} /> {warn}
          </div>
        )}
      </span>
    </div>
  )
}
