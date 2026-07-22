import { useEffect, useState } from 'react'
import { endpoints, errorMessage } from '../../api/client'
import type { SettingsDump } from '../../api/types'
import { Button } from '../../components/Button'
import { GearIcon } from '../../components/icons'
import { EmptyState, Spinner } from '../../components/Misc'
import { OpenSearchDownModal } from '../../components/OpenSearchDownModal'
import { QdrantDownModal } from '../../components/QdrantDownModal'
import { PageHeader } from '../../components/PageHeader'
import { Pill } from '../../components/Pill'
import { toast } from '../../store/toastStore'
import { SectionCard } from './SectionCard'
import { WiringBanner } from './WiringBanner'
import { KNOWN_KEYS, REINDEX_KEYS, SECTIONS, type FieldDef } from './schema'

type Current = SettingsDump

// Only relevant for restart_mode=reload (uvicorn --reload in local dev): every other mode applies
// live in-process, with no worker respawn and thus nothing to poll for -- see /api/v1/settings.
async function waitForRestart(onTick: (msg: string) => void) {
  const deadline = Date.now() + 30000
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 1000))
    try {
      const h = await endpoints.health()
      if (h.status === 'ok') return
    } catch {
      // worker still down, keep polling
    }
  }
  onTick('Service did not come back within 30s — check the server, then reload.')
}

export function SettingsPage() {
  const [initial, setInitial] = useState<Current | null>(null)
  const [current, setCurrent] = useState<Current | null>(null)
  const [loadError, setLoadError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')
  const [showOsModal, setShowOsModal] = useState(false)
  const [showQdrantModal, setShowQdrantModal] = useState(false)
  const [pendingChanges, setPendingChanges] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    endpoints
      .settings()
      .then((s) => {
        setInitial(s)
        setCurrent(s)
      })
      .catch((e) => setLoadError(errorMessage(e)))
  }, [])

  function handleChange(key: string, value: string | number | boolean) {
    if (!current || !initial) return
    setCurrent({ ...current, [key]: value })
    if (REINDEX_KEYS.has(key) && value !== initial[key]) {
      toast('Embedding changed — create a NEW index and re-ingest; existing indices are locked to their build model.', 'error')
    }
  }

  async function onSave() {
    if (!current || !initial) return
    const changes: Record<string, unknown> = {}
    for (const key of Object.keys(current)) if (current[key] !== initial[key]) changes[key] = current[key]
    if (!Object.keys(changes).length) {
      setSaveStatus('No changes.')
      return
    }
    // Switching to OpenSearch / Qdrant doesn't mean it's actually running -- they're opt-in behind
    // compose profiles. Check first so a broken retrieval call isn't the first sign anything's wrong.
    if (changes.vector_store === 'opensearch') {
      const reachable = await endpoints.opensearchStatus().then((s) => s.reachable).catch(() => false)
      if (!reachable) {
        setPendingChanges(changes)
        setShowOsModal(true)
        return
      }
    }
    if (changes.vector_store === 'qdrant') {
      const reachable = await endpoints.qdrantStatus().then((s) => s.reachable).catch(() => false)
      if (!reachable) {
        setPendingChanges(changes)
        setShowQdrantModal(true)
        return
      }
    }
    await commitSave(changes)
  }

  async function commitSave(changes: Record<string, unknown>) {
    setShowOsModal(false)
    setShowQdrantModal(false)
    setSaving(true)
    setSaveStatus('Saving…')
    try {
      const res = await endpoints.saveSettings(changes)
      if (res.restarting) {
        // dev --reload only: the worker is respawning, so wait for it to answer again.
        setSaveStatus('Restarting service…')
        await waitForRestart(setSaveStatus)
      }
      setSaveStatus('')
      const fresh = await endpoints.settings()
      setInitial(fresh)
      setCurrent(fresh)
      const st = await endpoints.onboardingStatus().catch(() => null)
      if (st && !st.configured) {
        toast(
          'Saved, but the service is not fully ready — check required provider fields (e.g. embedding model). You will stay in the console on refresh.',
          'error',
        )
      } else {
        toast(res.restarting ? 'Settings saved and service restarted' : 'Settings saved and applied — no restart needed', 'success')
      }
    } catch (e) {
      setSaveStatus(`Save failed: ${errorMessage(e)}`)
    } finally {
      setSaving(false)
    }
  }

  if (loadError) return <div className="text-bad">Could not load settings: {loadError}</div>
  if (!current || !initial) {
    return (
      <EmptyState>
        <Spinner />
      </EmptyState>
    )
  }

  const unmapped = Object.keys(initial).filter((k) => !KNOWN_KEYS.has(k)).sort()
  const sections = unmapped.length
    ? [
        ...SECTIONS,
        {
          id: 'other',
          title: 'Other (unmapped)',
          fields: unmapped.map((k) => ({
            key: k,
            type: (typeof initial[k] === 'number' ? 'number' : typeof initial[k] === 'boolean' ? 'bool' : 'text') as FieldDef['type'],
          })),
        },
      ]
    : SECTIONS

  return (
    <div>
      <PageHeader icon={GearIcon} title="Settings">
        Editable mirror of <code className="bg-surface-2 px-1.5 py-0.5 rounded">.env</code>, grouped by module. Inactive blocks are dimmed;{' '}
        <Pill tone="warn">needs value</Pill> marks required credentials still empty. Save writes{' '}
        <code className="bg-surface-2 px-1.5 py-0.5 rounded">.env</code> and applies immediately — no restart needed.
      </PageHeader>
      <WiringBanner current={current} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-4">
        {sections.map((s) => (
          <SectionCard key={s.id} section={s} current={current} initial={initial} onChange={handleChange} />
        ))}
      </div>
      <div className="flex gap-4 items-center mt-2">
        <Button variant="primary" onClick={onSave} disabled={saving}>
          {saving && <Spinner />} Save
        </Button>
        <span className="text-xs text-ink-faint">{saveStatus}</span>
      </div>
      <OpenSearchDownModal
        open={showOsModal}
        onClose={() => setShowOsModal(false)}
        onReachable={() => pendingChanges && commitSave(pendingChanges)}
        onContinueAnyway={() => pendingChanges && commitSave(pendingChanges)}
      />
      <QdrantDownModal
        open={showQdrantModal}
        onClose={() => setShowQdrantModal(false)}
        onReachable={() => pendingChanges && commitSave(pendingChanges)}
        onContinueAnyway={() => pendingChanges && commitSave(pendingChanges)}
      />
    </div>
  )
}
