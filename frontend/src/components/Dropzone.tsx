import { useRef, useState } from 'react'
import { UploadIcon } from './icons'

export function Dropzone({ onFiles, hint = 'Click to choose files, or drag them here' }: { onFiles: (files: FileList) => void; hint?: string }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [label, setLabel] = useState('')

  function describe(files: FileList) {
    if (files.length > 1) return `${files.length} files: ${[...files].map((f) => f.name).join(', ')}`
    return files[0]?.name ?? ''
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
        if (e.dataTransfer.files.length) {
          setLabel(describe(e.dataTransfer.files))
          onFiles(e.dataTransfer.files)
        }
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
          if (e.target.files?.length) {
            setLabel(describe(e.target.files))
            onFiles(e.target.files)
          }
        }}
      />
    </div>
  )
}
