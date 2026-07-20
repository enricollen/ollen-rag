import type { FilterSpec } from '../api/types'
import { Button } from './Button'
import { Select, TextInput } from './Field'

const OPERATORS = ['==', '!=', '>', '>=', '<', '<=', 'in', 'nin']

export interface FilterRow {
  key: string
  value: string
  operator: string
}

export function emptyFilterRow(): FilterRow {
  return { key: '', value: '', operator: '==' }
}

export function filterRowsToSpecs(rows: FilterRow[]): FilterSpec[] {
  return rows
    .filter((r) => r.key.trim())
    .map((r) => {
      let value: unknown = r.value
      try {
        value = JSON.parse(r.value)
      } catch {
        /* keep as string */
      }
      return { key: r.key.trim(), value, operator: r.operator }
    })
}

export function FilterBuilder({
  rows,
  onChange,
  condition,
  onConditionChange,
}: {
  rows: FilterRow[]
  onChange: (rows: FilterRow[]) => void
  condition: 'and' | 'or'
  onConditionChange: (c: 'and' | 'or') => void
}) {
  function update(i: number, patch: Partial<FilterRow>) {
    onChange(rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r)))
  }
  function remove(i: number) {
    onChange(rows.filter((_, idx) => idx !== i))
  }
  return (
    <div>
      {rows.map((row, i) => (
        <div key={i} className="flex gap-2 mb-2 items-center">
          <TextInput
            placeholder="key (e.g. bucket)"
            value={row.key}
            onChange={(e) => update(i, { key: e.target.value })}
            className="flex-1"
          />
          <Select value={row.operator} onChange={(e) => update(i, { operator: e.target.value })} className="w-[90px] flex-none">
            {OPERATORS.map((o) => (
              <option key={o}>{o}</option>
            ))}
          </Select>
          <TextInput placeholder="value" value={row.value} onChange={(e) => update(i, { value: e.target.value })} className="flex-1" />
          <Button type="button" variant="secondary" onClick={() => remove(i)} className="flex-none px-2.5">
            &times;
          </Button>
        </div>
      ))}
      <div className="flex gap-2 items-center mt-1">
        <Button type="button" variant="secondary" onClick={() => onChange([...rows, emptyFilterRow()])}>
          + filter
        </Button>
        <Select value={condition} onChange={(e) => onConditionChange(e.target.value as 'and' | 'or')} className="max-w-[110px]">
          <option value="and">AND</option>
          <option value="or">OR</option>
        </Select>
      </div>
    </div>
  )
}
