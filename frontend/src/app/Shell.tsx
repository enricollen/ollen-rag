import { useEffect, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { endpoints } from '../api/client'
import { AlertTriangleIcon } from '../components/icons'
import { ActiveConfigBanner } from './ActiveConfigBanner'
import { Sidebar } from './Sidebar'

// Operator-console layout used by every route except /welcome.
// - needs_wizard (no LLM provider yet) -> bounce to the first-run wizard
// - configured=false after a settings edit -> stay here; show a soft banner (do NOT re-wizard)
export function Shell() {
  const location = useLocation()
  const navigate = useNavigate()
  const [notReady, setNotReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    endpoints
      .onboardingStatus()
      .then((st) => {
        if (cancelled) return
        if (st.needs_wizard) {
          navigate('/welcome', { replace: true })
          return
        }
        setNotReady(!st.configured)
      })
      .catch(() => {
        /* status unreachable -- let the page's own error state show instead */
      })
    return () => {
      cancelled = true
    }
  }, [location.pathname, navigate])

  return (
    <div className="flex min-h-screen scope-grid">
      <div className="grain-overlay" />
      <Sidebar />
      <main className="flex-1 px-10 py-8 pb-16 max-w-[1180px]">
        {notReady && (
          <div className="mb-4 flex items-start gap-2.5 rounded-control border border-warn/40 bg-warn/10 px-3.5 py-2.5 text-sm text-ink">
            <AlertTriangleIcon size={16} className="text-warn flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-medium text-warn">Service not fully ready</div>
              <p className="text-xs text-ink-dim m-0 mt-0.5 leading-relaxed">
                A required provider setting is incomplete (often the embedding model after switching
                providers). Fix it in{' '}
                <button
                  type="button"
                  className="text-accent underline underline-offset-2 hover:text-accent-strong"
                  onClick={() => navigate('/settings')}
                >
                  Settings
                </button>
                — you will not be sent back through onboarding.
              </p>
            </div>
          </div>
        )}
        <ActiveConfigBanner refreshKey={location.pathname} />
        <Outlet />
      </main>
    </div>
  )
}
