import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, the UI runs on :5173 and proxies API calls to the FastAPI server on :8000.
// In production, `npm run build` writes ui/dist and FastAPI serves it at "/".
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/claims': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
