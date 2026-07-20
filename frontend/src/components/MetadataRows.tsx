import { Button } from './Button'
import { TextInput } from './Field'

export interface MetaRow {
  key: string
  value: string
}

export function metaRowsToObject(rows: MetaRow[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (const r of rows) if (r.key.trim()) out[r.key.trim()] = r.value
  return out
}

export function MetadataRows({ rows, onChange }: { rows: MetaRow[]; onChange: (rows: MetaRow[]) => void }) {
  function update(i: number, patch: Partial<MetaRow>) {
    onChange(rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r)))
  }
  function remove(i: number) {
    onChange(rows.filter((_, idx) => idx !== i))
  }
  return (
    <div>
      {rows.map((row, i) => (
        <div key={i} className="flex gap-2 mb-2">
          <TextInput placeholder="key" value={row.key} onChange={(e) => update(i, { key: e.target.value })} className="flex-1" />
          <TextInput placeholder="value" value={row.value} onChange={(e) => update(i, { value: e.target.value })} className="flex-1" />
          <Button type="button" variant="secondary" onClick={() => remove(i)} className="flex-none px-2.5">
            &times;
          </Button>
        </div>
      ))}
      <Button type="button" variant="secondary" onClick={() => onChange([...rows, { key: '', value: '' }])}>
        + metadata field
      </Button>
    </div>
  )
}
