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

export interface ModalityChoice {
  id: string
  label: string
  description: string
  /** true when no credentials are needed to use this provider. */
  keyless: boolean
  fields: CredField[]
}

// provider choices offered in step 1, with the settings each one needs. mirrors the old wizard's
// llm_choices (ui/pages/onboarding.js) plus a short description for the new card layout.
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

// defaults when the operator skips these optional steps — local, keyless, already the settings
// defaults. written explicitly so a virgin .env gets a concrete provider rather than "".
export const DEFAULT_EMBEDDING_PROVIDER = 'fastembed'
export const DEFAULT_RERANKER_PROVIDER = 'sentence-transformers'

export const EMBEDDING_CHOICES: ModalityChoice[] = [
  {
    id: 'fastembed',
    label: 'Local (fastembed)',
    description: 'Runs on this machine — recommended if you skip.',
    keyless: true,
    fields: [],
  },
  {
    id: 'litellm-ollama',
    label: 'Ollama',
    description: 'Local embedding models via Ollama.',
    keyless: true,
    fields: [{ key: 'ollama_embedding_model', label: 'Model (e.g. nomic-embed-text)' }],
  },
  {
    id: 'watsonx',
    label: 'watsonx.ai',
    description: "IBM's embedding models.",
    keyless: false,
    fields: [
      { key: 'watsonx_url', label: 'watsonx URL' },
      { key: 'watsonx_apikey', label: 'API key', secret: true },
      { key: 'watsonx_project_id', label: 'Project ID', secret: true },
      { key: 'watsonx_embedding_model_id', label: 'Model (e.g. ibm/slate-125m-english-rtrvr)' },
    ],
  },
  {
    id: 'litellm-openai',
    label: 'OpenAI',
    description: 'text-embedding models via your OpenAI API key.',
    keyless: false,
    fields: [
      { key: 'openai_embedding_model', label: 'Model (e.g. text-embedding-3-small)' },
      { key: 'openai_api_key', label: 'API key', secret: true },
    ],
  },
  {
    id: 'litellm-openrouter',
    label: 'OpenRouter',
    description: 'Hosted embeddings through OpenRouter.',
    keyless: false,
    fields: [
      { key: 'openrouter_embedding_model', label: 'Model' },
      { key: 'openrouter_api_key', label: 'API key', secret: true },
    ],
  },
  {
    id: 'litellm',
    label: 'Other (LiteLLM)',
    description: 'Any LiteLLM embedding endpoint (Cohere, Groq, …).',
    keyless: false,
    fields: [
      { key: 'litellm_embedding_model', label: 'Model string' },
      { key: 'litellm_embedding_api_key', label: 'API key', secret: true },
      { key: 'litellm_embedding_api_base', label: 'API base (optional)' },
    ],
  },
  {
    id: 'litellm-watsonx',
    label: 'watsonx via LiteLLM',
    description: 'watsonx embeddings through the LiteLLM adapter.',
    keyless: false,
    fields: [
      { key: 'watsonx_url', label: 'watsonx URL' },
      { key: 'watsonx_apikey', label: 'API key', secret: true },
      { key: 'watsonx_project_id', label: 'Project ID', secret: true },
      { key: 'watsonx_embedding_model_id', label: 'Model' },
    ],
  },
]

export const RERANKER_CHOICES: ModalityChoice[] = [
  {
    id: 'sentence-transformers',
    label: 'Local (cross-encoder)',
    description: 'Runs on this machine — recommended if you skip.',
    keyless: true,
    fields: [],
  },
  {
    id: 'litellm-watsonx',
    label: 'watsonx.ai',
    description: 'Rerank via watsonx through LiteLLM.',
    keyless: false,
    fields: [
      { key: 'watsonx_url', label: 'watsonx URL' },
      { key: 'watsonx_apikey', label: 'API key', secret: true },
      { key: 'watsonx_project_id', label: 'Project ID', secret: true },
      { key: 'watsonx_reranker_model_id', label: 'Model' },
    ],
  },
  {
    id: 'litellm',
    label: 'Other (LiteLLM)',
    description: 'Cohere, Jina, or any LiteLLM rerank endpoint.',
    keyless: false,
    fields: [
      { key: 'litellm_rerank_model', label: 'Model (e.g. cohere/rerank-v3.5)' },
      { key: 'litellm_rerank_api_key', label: 'API key', secret: true },
      { key: 'litellm_rerank_api_base', label: 'API base (optional)' },
    ],
  },
]
