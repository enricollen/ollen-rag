import type { ChunkingConfig } from '../api/types'

// One-line summary of a chunking config dict {strategy, ...knobs}.
export function chunkingSummary(chunking: ChunkingConfig | null | undefined): string {
  if (!chunking) return 'unrecorded (legacy index)'
  const knobs = Object.entries(chunking)
    .filter(([k]) => k !== 'strategy')
    .map(([k, v]) => `${k}=${v}`)
  return [chunking.strategy, ...knobs].join(' · ')
}

// Wizard "Test connection" probes surface raw provider/SDK exception text (DNS errors, httpx
// connection errors, auth JSON blobs, tracebacks). Recognize the common failure shapes and turn
// them into a short, actionable hint; anything unrecognized is trimmed rather than dumped raw.
const _CONNECTION_ERROR_RULES: [RegExp, string][] = [
  [/name resolution|nodename nor servname|getaddrinfo|ENOTFOUND/i, "Can't reach that host \u2014 check the URL/endpoint is correct."],
  [/connection refused|ECONNREFUSED|failed to establish a new connection/i, 'Connection refused \u2014 is the service running at that address?'],
  [/timed out|timeout|ETIMEDOUT/i, 'Connection timed out \u2014 check the URL and network/firewall.'],
  [/401|unauthorized|invalid[_ ]api[_ ]?key|authentication/i, 'Authentication failed \u2014 check your API key.'],
  [/403|forbidden/i, 'Access denied \u2014 check your API key permissions.'],
  [/404|not found/i, 'Not found \u2014 check the model name, endpoint, or project ID.'],
  [/429|rate limit/i, 'Rate limited \u2014 too many requests, try again shortly.'],
  [/ssl|certificate/i, 'TLS/certificate error \u2014 check the URL (http vs https) or certificate settings.'],
]

export function friendlyConnectionError(raw: string): string {
  const flat = raw.replace(/\s+/g, ' ').trim()
  for (const [pattern, hint] of _CONNECTION_ERROR_RULES) {
    if (pattern.test(flat)) return hint
  }
  const short = flat.length > 140 ? `${flat.slice(0, 140)}\u2026` : flat
  return short || 'Connection test failed.'
}

// Validated dark-mode categorical palette, fixed order (never cycled) so color maps to the same
// identity across every chart in the app.
export const CHART_PALETTE = ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926']
export const CHART_GRID = 'rgba(255,255,255,0.08)'
