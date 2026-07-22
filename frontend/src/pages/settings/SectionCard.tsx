import type { SettingsDump } from '../../api/types'
import { Panel } from '../../components/Panel'
import { Pill } from '../../components/Pill'
import { FieldRow } from './FieldRow'
import { selectionOf, type SectionDef } from './schema'

type Current = SettingsDump

export function SectionCard({
  section,
  current,
  initial,
  onChange,
}: {
  section: SectionDef
  current: Current
  initial: Current
  onChange: (key: string, value: string | number | boolean) => void
}) {
  const active = !section.gate || section.gate(selectionOf(current))
  return (
    <Panel
      accent={section.control}
      dim={!section.control && !active}
      title={section.title}
      badge={!section.control && <Pill tone={active ? 'ok' : 'default'}>{active ? 'active' : 'inactive'}</Pill>}
    >
      {section.note && <div className="text-xs text-ink-faint -mt-2 mb-2.5">{section.note}</div>}
      <div>
        {section.fields.map((f) => (
          <FieldRow key={f.key} field={f} current={current} initial={initial} sectionActive={active} onChange={onChange} />
        ))}
      </div>
      {(section.id === 'vs_chroma' || section.id === 'vs_os' || section.id === 'vs_qdrant') && (
        <div className="text-xs text-ink-faint mt-2">Switching vector store does not migrate data &mdash; each store holds its own indices.</div>
      )}
    </Panel>
  )
}
