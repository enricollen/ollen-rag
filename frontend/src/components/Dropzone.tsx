import { useRef, useState } from 'react'
import { UploadIcon } from './icons'

// always hand callers a real File[] — dataTransfer.FileList can go stale after the drop
// event ends, which makes xhr FormData uploads fail with a generic "Network error".
export function Dropzone({
  onFiles,
  hint = 'Click to choose files, or drag them here',
}: {
  onFiles: (files: File[]) => void
  hint?: string
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [label, setLabel] = useState('')

  function describe(files: File[]) {
    if (files.length > 1) return `${files.length} files: ${files.map((f) => f.name).join(', ')}`
    return files[0]?.name ?? ''
  }

  function take(list: FileList | null | undefined) {
    if (!list?.length) return
    const files = Array.from(list)
    setLabel(describe(files))
    onFiles(files)
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        take(e.dataTransfer.files)
      }}
      className={`border-[1.5px] border-dashed rounded-panel p-8 text-center cursor-pointer transition-colors ${
        dragging ? 'border-accent bg-accent-soft' : 'border-line hover:border-accent/60'
      }`}
    >
      <UploadIcon size={24} className={`mx-auto mb-2 ${dragging ? 'text-accent' : 'text-ink-faint'}`} />
      <div className="text-sm text-ink-dim">{hint}</div>
      {label && <div className="font-semibold text-signal mt-1.5 text-sm">{label}</div>}
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          take(e.target.files)
          // allow re-selecting the same files later
          e.target.value = ''
        }}
      />
    </div>
  )
}
