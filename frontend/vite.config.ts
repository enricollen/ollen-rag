import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// served by FastAPI's StaticFiles mount at /ui/ (see backend/app.py) -- base must match that
// prefix so built asset URLs resolve correctly both in docker and in `npm run preview`.
export default defineConfig({
  base: '/ui/',
  plugins: [react(), tailwindcss()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/ready': 'http://localhost:8000',
    },
  },
})
