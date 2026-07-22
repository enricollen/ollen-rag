import { useMemo } from 'react'
import type { ConfigResponse } from '../api/types'
import { Select } from './Field'

export interface RerankerSelection {
  reranker_provider: string | null
  reranker_model: string | null
}

// A provider with an empty model list is free-form (e.g. the generic "litellm" provider); it can
// only offer whatever model is configured for it, so it contributes an option only when
// reranker_default_models names one. The <select>'s value packs "provider::model" since model ids
// contain "/", so any other delimiter scheme would be ambiguous.
export function RerankerSelect({
  cfg,
  value,
  onChange,
}: {
  cfg: Pick<ConfigResponse, 'reranker_model_choices' | 'reranker_default_models' | 'reranker_provider' | 'reranker_model'>
  value: RerankerSelection
  onChange: (sel: RerankerSelection) => void
}) {
  const choices = cfg.reranker_model_choices || {}
  const defaults = cfg.reranker_default_models || {}

  const groups = useMemo(
    () =>
      Object.entries(choices)
        .map(([provider, models]) => {
          const options = models && models.length ? models : [defaults[provider]].filter(Boolean)
          return { provider, options }
        })
        .filter((g) => g.options.length),
    [choices, defaults],
  )

  const selectedValue =
    value.reranker_provider && value.reranker_model
      ? `${value.reranker_provider}::${value.reranker_model}`
      : cfg.reranker_provider && cfg.reranker_model
        ? `${cfg.reranker_provider}::${cfg.reranker_model}`
        : ''

  return (
    <Select
      value={selectedValue}
      onChange={(e) => {
        const [provider, model] = e.target.value.split('::')
        onChange({ reranker_provider: provider || null, reranker_model: model || null })
      }}
    >
      {groups.map((g) => (
        <optgroup key={g.provider} label={g.provider}>
          {g.options.map((m) => (
            <option key={m} value={`${g.provider}::${m}`}>
              {m}
            </option>
          ))}
        </optgroup>
      ))}
    </Select>
  )
}
