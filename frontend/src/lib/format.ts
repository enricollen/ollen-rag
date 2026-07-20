import type { ChunkingConfig } from '../api/types'

// One-line summary of a chunking config dict {strategy, ...knobs}.
export function chunkingSummary(chunking: ChunkingConfig | null | undefined): string {
  if (!chunking) return 'unrecorded (legacy index)'
  const knobs = Object.entries(chunking)
    .filter(([k]) => k !== 'strategy')
    .map(([k, v]) => `${k}=${v}`)
  return [chunking.strategy, ...knobs].join(' · ')
}

// Validated dark-mode categorical palette, fixed order (never cycled) so color maps to the same
// identity across every chart in the app.
export const CHART_PALETTE = ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926']
export const CHART_GRID = 'rgba(255,255,255,0.08)'
