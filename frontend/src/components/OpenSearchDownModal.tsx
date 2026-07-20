import { useState } from 'react'
import { endpoints } from '../api/client'
import { Button } from './Button'
import { AlertTriangleIcon, CheckIcon, CopyIcon, TerminalIcon } from './icons'
import { Modal } from './Modal'

const START_COMMAND = 'docker compose --profile opensearch up -d'

/** Shown when OpenSearch is selected but not reachable -- OpenSearch is opt-in behind compose's
 * `opensearch` profile (see docker-compose.yml), so it's simply not running until that profile is
 * used at least once. Offers the exact command to run plus a re-check, rather than the app trying
 * to start a sibling container itself. */
export function OpenSearchDownModal({
  open,
  onClose,
  onReachable,
  onContinueAnyway,
}: {
  open: boolean
  onClose: () => void
  onReachable: () => void
  onContinueAnyway: () => void
}) {
  const [checking, setChecking] = useState(false)
  const [copied, setCopied] = useState(false)
  const [stillDown, setStillDown] = useState(false)

  async function recheck() {
    setChecking(true)
    setStillDown(false)
    try {
      const { reachable } = await endpoints.opensearchStatus()
      if (reachable) onReachable()
      else setStillDown(true)
    } catch {
      setStillDown(true)
    } finally {
      setChecking(false)
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(START_COMMAND)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard permission denied -- the command is still selectable/visible, no big deal
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="OpenSearch isn't running" icon={<AlertTriangleIcon size={16} className="text-warn" />}>
      <p className="text-xs text-ink-dim mb-3 leading-relaxed">
        OpenSearch sits behind an opt-in compose profile, so it doesn't start with a plain{' '}
        <code className="bg-surface-2 px-1 py-0.5 rounded">docker compose up</code>. Run this once in the same
        directory as your <code className="bg-surface-2 px-1 py-0.5 rounded">docker-compose.yml</code>:
      </p>
      <div className="flex items-center gap-2 rounded-control border border-line bg-surface-2/70 px-3 py-2 mb-4">
        <TerminalIcon size={14} className="text-ink-faint flex-shrink-0" />
        <code className="text-xs text-ink flex-1 overflow-x-auto whitespace-nowrap">{START_COMMAND}</code>
        <button
          type="button"
          onClick={copy}
          title="Copy command"
          className="text-ink-faint hover:text-accent transition-colors flex-shrink-0"
        >
          {copied ? <CheckIcon size={14} className="text-good" /> : <CopyIcon size={14} />}
        </button>
      </div>
      {stillDown && <p className="text-xs text-bad mb-3">Still not reachable -- give it a few seconds to boot, then try again.</p>}
      <div className="flex items-center justify-between gap-2">
        <Button variant="ghost" onClick={onContinueAnyway} className="text-xs">
          Continue anyway
        </Button>
        <Button variant="primary" onClick={recheck} disabled={checking} className="text-xs">
          {checking ? 'Checking…' : 'Check again'}
        </Button>
      </div>
    </Modal>
  )
}
