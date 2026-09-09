import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // The dashboard source is kept alongside the Python pipeline, one level
    // above this Vite project. Resolve its package imports here.
    alias: {
      react: fileURLToPath(new URL('./node_modules/react', import.meta.url)),
      recharts: fileURLToPath(new URL('./node_modules/recharts', import.meta.url)),
    },
  },
  server: {
    fs: { allow: ['..'] },
  },
})
