import { useEffect, useMemo } from 'react'
import type { IndexListEntry } from '../api/types'
import { Select } from './Field'

// Picks the index with the MOST documents as the default -- avoids an empty/stray index (which
// sorts first in _cat output) silently hijacking the default selection.
export function defaultIndexName(indices: IndexListEntry[]): string {
  if (!indices.length) return ''
  const top = indices.reduce((a, b) => (Number(b['docs.count']) > Number(a['docs.count']) ? b : a))
  return top.index
}

export function IndexSelect({
  indices,
  value,
  onChange,
}: {
  indices: IndexListEntry[]
  value: string
  onChange: (v: string) => void
}) {
  const defaultName = useMemo(() => defaultIndexName(indices), [indices])

  // Keep parent state in sync with the visible default. Showing `value || defaultName` in the
  // <select> without calling onChange left Visualizer (and similar) with selected="" -- so the
  // dropdown looked filled while the chart/fetch gated on `selected` rendered nothing.
  useEffect(() => {
    if (!value && defaultName) onChange(defaultName)
  }, [value, defaultName, onChange])

  return (
    <Select value={value || defaultName} onChange={(e) => onChange(e.target.value)}>
      {!indices.length && <option value="">no indices yet</option>}
      {indices.map((ix) => (
        <option key={ix.index} value={ix.index}>
          {ix.index} ({ix['docs.count']} docs)
        </option>
      ))}
    </Select>
  )
}
