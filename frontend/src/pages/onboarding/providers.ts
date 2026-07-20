export interface CredField {
  key: string
  label: string
  secret?: boolean
}

export interface LlmChoice {
  id: string
  label: string
  description: string
  fields: CredField[]
}

// Provider choices offered in step 1, with the settings each one needs. Mirrors the old wizard's
// LLM_CHOICES (ui/pages/onboarding.js) plus a short description for the new card layout.
export const LLM_CHOICES: LlmChoice[] = [
  {
    id: 'litellm-ollama',
    label: 'Local (Ollama)',
    description: 'No account needed — runs entirely on this machine.',
    fields: [],
  },
  {
    id: 'watsonx',
    label: 'watsonx.ai',
    description: "IBM's enterprise LLM platform.",
    fields: [
      { key: 'watsonx_url', label: 'watsonx URL' },
      { key: 'watsonx_apikey', label: 'API key', secret: true },
      { key: 'watsonx_project_id', label: 'Project ID', secret: true },
    ],
  },
  {
    id: 'litellm-openai',
    label: 'OpenAI',
    description: 'GPT models via your OpenAI API key.',
    fields: [
      { key: 'openai_model', label: 'Model (e.g. gpt-4o-mini)' },
      { key: 'openai_api_key', label: 'API key', secret: true },
    ],
  },
  {
    id: 'litellm-openrouter',
    label: 'OpenRouter',
    description: 'One key, access to many hosted models.',
    fields: [
      { key: 'openrouter_model', label: 'Model' },
      { key: 'openrouter_api_key', label: 'API key', secret: true },
    ],
  },
  {
    id: 'litellm',
    label: 'Other (LiteLLM)',
    description: 'Point at any LiteLLM-compatible endpoint.',
    fields: [
      { key: 'litellm_model', label: 'Model string' },
      { key: 'litellm_api_key', label: 'API key', secret: true },
      { key: 'litellm_api_base', label: 'API base (optional)' },
    ],
  },
]
