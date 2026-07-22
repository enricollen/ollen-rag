import { useEffect, useState } from 'react'
import { endpoints } from '../api/client'

export interface HealthState {
  status: 'ok' | 'bad' | 'pending'
  strategiesLabel: string
}

// Polls /health + /api/v1/strategies every 15s, mirroring the old console's sidebar footer.
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ status: 'pending', strategiesLabel: 'strategies: …' })

  useEffect(() => {
    let cancelled = false
    async function tick() {
      let status: HealthState['status'] = 'bad'
      try {
        await endpoints.health()
        status = 'ok'
      } catch {
        status = 'bad'
      }
      let strategiesLabel = 'strategies: unknown'
      try {
        const s = await endpoints.strategies()
        strategiesLabel = `strategies: ${s.strategies.join(', ')}`
      } catch {
        /* keep fallback label */
      }
      if (!cancelled) setState({ status, strategiesLabel })
    }
    tick()
    const id = setInterval(tick, 15000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return state
}
