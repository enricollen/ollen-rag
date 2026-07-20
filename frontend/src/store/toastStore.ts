import { create } from 'zustand'

export type ToastKind = 'info' | 'error' | 'success'

export interface ToastItem {
  id: number
  message: string
  kind: ToastKind
}

let nextId = 1

interface ToastState {
  toasts: ToastItem[]
  push: (message: string, kind?: ToastKind) => void
  dismiss: (id: number) => void
}

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (message, kind = 'info') => {
    const id = nextId++
    set((s) => ({ toasts: [...s.toasts, { id, message, kind }] }))
    setTimeout(() => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })), 5000)
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}))

// Standalone helper for non-component call sites (mirrors the old ui/lib.js toast()).
export function toast(message: string, kind: ToastKind = 'info') {
  useToastStore.getState().push(message, kind)
}
