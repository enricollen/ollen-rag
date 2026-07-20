import { Pill } from './Pill'

// Row of one-click pills for buckets used earlier this session (from historyStore), shown under
// a bucket text field so a repeat bucket name doesn't have to be retyped. Used by both the
// "create new index" and "add to existing index" ingestion flows.
export function BucketQuickPick({ buckets, onPick }: { buckets: string[]; onPick: (bucket: string) => void }) {
  if (!buckets.length) return null
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {buckets.map((b) => (
        <Pill key={b} onClick={() => onPick(b)}>
          {b}
        </Pill>
      ))}
    </div>
  )
}
