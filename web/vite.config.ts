import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Fixed port: the API's CORS_ORIGINS allows http://localhost:5173 only.
  server: { port: 5173, strictPort: true },
})
