import { NavLink } from 'react-router-dom'
import { StatusDot } from '../components/Misc'
import { useHealth } from './useHealth'
import { ThemeToggle } from './ThemeToggle'

const ICON_CLS = 'w-[18px] h-[18px] flex-shrink-0 opacity-85'

const NAV = [
  {
    to: '/settings',
    label: 'Settings',
    icon: (
      <path
        fill="currentColor"
        d="M19.4 13a7.5 7.5 0 0 0 .07-1 7.5 7.5 0 0 0-.07-1l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.4.96a7.4 7.4 0 0 0-1.7-1L14.5 2.6a.5.5 0 0 0-.5-.4h-3.84a.5.5 0 0 0-.5.4l-.43 2.58a7.4 7.4 0 0 0-1.7 1l-2.4-.96a.5.5 0 0 0-.6.22L2.6 8.76a.5.5 0 0 0 .12.64L4.75 11a7.5 7.5 0 0 0 0 2l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32c.14.24.42.33.66.22l2.4-.96c.53.43 1.1.77 1.7 1l.43 2.58c.05.24.26.4.5.4h3.84c.24 0 .45-.16.5-.4l.43-2.58c.6-.23 1.17-.57 1.7-1l2.4.96c.24.1.52 0 .66-.22l1.92-3.32a.5.5 0 0 0-.12-.64L19.4 13ZM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7Z"
      />
    ),
  },
  {
    to: '/indices',
    label: 'Indices',
    icon: <path fill="currentColor" d="M4 4h16v4H4V4Zm0 6h16v4H4v-4Zm0 6h16v4H4v-4Z" />,
  },
  {
    to: '/ingestion',
    label: 'Ingestion KB',
    icon: (
      <path
        fill="currentColor"
        d="M12 2 3 7v10l9 5 9-5V7l-9-5Zm0 2.2 6.9 3.83L12 11.87 5.1 8.03 12 4.2ZM5 9.7l6 3.33v6.86l-6-3.33V9.7Zm8 10.19v-6.86l6-3.33v6.86l-6 3.33Z"
      />
    ),
  },
  {
    to: '/retrieval',
    label: 'Retrieval',
    icon: (
      <path
        fill="currentColor"
        d="M10 2a8 8 0 1 0 4.9 14.32l5.39 5.39 1.4-1.42-5.38-5.38A8 8 0 0 0 10 2Zm0 2a6 6 0 1 1 0 12 6 6 0 0 1 0-12Z"
      />
    ),
  },
  {
    to: '/query',
    label: 'Query (e2e)',
    icon: (
      <path
        fill="currentColor"
        d="M4 4h16a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H8l-4.5 4.2A.5.5 0 0 1 3 20.85V5a1 1 0 0 1 1-1Zm2 4v2h12V8H6Zm0 4v2h8v-2H6Z"
      />
    ),
  },
  {
    to: '/eval',
    label: 'Eval',
    icon: (
      <path
        fill="currentColor"
        d="M5 3h14a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Zm2 4v2h10V7H7Zm0 4v2h10v-2H7Zm0 4v2h6v-2H7Z"
      />
    ),
  },
]

export function Sidebar() {
  const health = useHealth()
  return (
    <aside className="w-[230px] flex-shrink-0 border-r border-line flex flex-col p-5 sticky top-0 h-screen bg-black/20">
      <div className="flex items-center gap-2.5 mb-8 px-1">
        <div
          className="neon-glow w-[34px] h-[34px] rounded-[9px] flex items-center justify-center font-bold text-[1rem] text-white"
          style={{ background: 'linear-gradient(135deg, var(--color-accent), var(--color-signal))' }}
        >
          N
        </div>
        <div>
          <div className="font-semibold text-[0.95rem] text-ink">ollen-rag</div>
          <div className="text-[0.72rem] text-ink-faint font-mono">service console</div>
        </div>
      </div>
      <nav className="flex flex-col gap-0.5 flex-1">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2.5 rounded-control text-[0.88rem] font-medium transition-colors ${
                isActive ? 'text-signal bg-signal-soft' : 'text-ink-dim hover:text-ink hover:bg-white/5'
              }`
            }
          >
            <svg viewBox="0 0 24 24" className={ICON_CLS}>
              {item.icon}
            </svg>
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-line pt-3 mt-2 flex flex-col gap-3">
        <ThemeToggle />
        <div>
          <div className="flex items-center gap-2 text-[0.8rem] text-ink-dim">
            <StatusDot status={health.status} />
            <span>{health.status === 'ok' ? 'service healthy' : health.status === 'bad' ? 'service unreachable' : 'checking…'}</span>
          </div>
          <div className="text-[0.72rem] text-ink-faint mt-1 pl-4 font-mono">{health.strategiesLabel}</div>
        </div>
      </div>
    </aside>
  )
}
