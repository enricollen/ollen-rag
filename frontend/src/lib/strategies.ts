// Chunking strategies shown as selectable cards in the create flow, each with a short
// description and a tiny illustrative split example (mark = chunk boundary). Ported from
// ui/ingest-common.js's STRATEGIES/CHUNK_FIELDS.

export interface StrategyDef {
  name: string
  desc: string
  example: string
  warn?: string
}

export const STRATEGIES: StrategyDef[] = [
  {
    name: 'sentence',
    desc: 'Splits on sentence boundaries. Simple, predictable, good default for prose.',
    example: 'The protocol assigns color codes.| Depending on severity, the codes are red and green.|',
  },
  {
    name: 'token',
    desc: 'Fixed-size token windows with overlap. Good for dense technical text without clear sentence structure.',
    example: '[ tok1 tok2 tok3 tok4 ]| [ tok3 tok4 tok5 tok6 ]|  (overlap = 2)',
  },
  {
    name: 'semantic',
    desc: 'Groups sentences by embedding similarity; splits where meaning shifts, not at a fixed size.',
    example: 'Triage assigns color codes. The codes indicate priority.| Neapolitan pizza uses flour and tomato.|',
  },
  {
    name: 'window',
    desc: 'One sentence per chunk, but stores surrounding sentences as context for the LLM at answer time.',
    example: 'node: "The codes indicate priority."  window: [prev, this, next]',
  },
  {
    name: 'llm',
    desc: 'Uses the LLM to judge topic boundaries between sentences — produces the most semantically coherent chunks. Slower than other strategies (one LLM call per boundary).',
    example: '"The protocol assigns color codes. The codes indicate priority."| "Neapolitan pizza uses flour…"|',
    warn: 'Significantly slower — one LLM call per sentence boundary.',
  },
]

export interface ChunkFieldDef {
  key: string
  label: string
  min?: number
  max?: number
  step?: number
}

// Which chunk knobs (Settings field names) are meaningful per strategy -- mirrors the server's
// CHUNK_PARAM_FIELDS. Only these are shown/sent so an index's recorded config stays clean.
export const CHUNK_FIELDS: Record<string, ChunkFieldDef[]> = {
  sentence: [
    { key: 'chunk_size', label: 'Chunk size (tokens)', min: 1, step: 1 },
    { key: 'chunk_overlap', label: 'Overlap (tokens)', min: 0, step: 1 },
  ],
  token: [
    { key: 'chunk_size', label: 'Chunk size (tokens)', min: 1, step: 1 },
    { key: 'chunk_overlap', label: 'Overlap (tokens)', min: 0, step: 1 },
  ],
  semantic: [{ key: 'semantic_breakpoint_percentile', label: 'Breakpoint percentile', min: 1, max: 100, step: 1 }],
  window: [{ key: 'sentence_window_size', label: 'Window size (sentences)', min: 1, step: 1 }],
  llm: [
    { key: 'llm_chunk_max_size', label: 'Max chunk size (tokens)', min: 1, step: 1 },
    { key: 'llm_chunk_window_size', label: 'Sentence window', min: 1, step: 1 },
  ],
}
