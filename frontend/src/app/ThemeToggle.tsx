import { MoonIcon, SunIcon } from '../components/icons'
import { useThemeStore } from '../store/themeStore'

// Compact pill switch, dark<->light, persisted via themeStore. Lives in the Sidebar footer.
export function ThemeToggle() {
  const theme = useThemeStore((s) => s.theme)
  const toggle = useThemeStore((s) => s.toggle)
  const isDark = theme === 'dark'

  return (
    <button
      type="button"
      onClick={toggle}
      title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      className="relative flex items-center w-full gap-2 rounded-control border border-line bg-surface-2/60 px-1 py-1 text-xs font-medium text-ink-dim hover:border-accent/40 transition-colors"
    >
      <span
        className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-[calc(var(--radius-control)-2px)] transition-colors ${
          isDark ? 'bg-accent text-white shadow-[var(--glow-accent)]' : ''
        }`}
      >
        <MoonIcon size={13} /> Dark
      </span>
      <span
        className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-[calc(var(--radius-control)-2px)] transition-colors ${
          !isDark ? 'bg-accent text-white shadow-[var(--glow-accent)]' : ''
        }`}
      >
        <SunIcon size={13} /> Light
      </span>
    </button>
  )
}
