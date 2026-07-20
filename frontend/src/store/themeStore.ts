import { create } from 'zustand'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'ollen-rag-theme'

function prefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches !== false
}

function readInitialTheme(): Theme {
  const stored = typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null
  if (stored === 'dark' || stored === 'light') return stored
  return prefersDark() ? 'dark' : 'light'
}

function applyTheme(theme: Theme) {
  document.documentElement.setAttribute('data-theme', theme)
}

interface ThemeState {
  theme: Theme
  toggle: () => void
  setTheme: (theme: Theme) => void
}

// Applied eagerly at module load (before React mounts) so there's no flash of the wrong theme.
const initial = readInitialTheme()
applyTheme(initial)

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: initial,
  toggle: () => get().setTheme(get().theme === 'dark' ? 'light' : 'dark'),
  setTheme: (theme) => {
    applyTheme(theme)
    window.localStorage.setItem(STORAGE_KEY, theme)
    set({ theme })
  },
}))
