import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
// `defineConfig` comes from vitest/config rather than vite: it is Vite's own,
// widened with the `test` key below. One config file rather than two keeps the
// `@` alias defined once — a second copy in a vitest.config.ts is the kind of
// duplicate that resolves correctly right up until someone changes one of them.
// Verified against the installed vitest 4.1.11, whose peer range is
// `vite: ^6 || ^7 || ^8` and which therefore accepts the installed Vite 8.2.1.
import { defineConfig } from 'vitest/config'

// Tailwind v4 is wired as a Vite plugin rather than through PostCSS: per the
// official "Using Vite" install guide there is no tailwind.config.js and no
// autoprefixer step — configuration is CSS-first, in src/styles/tailwind.css.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    // Pinned because the API whitelists exactly localhost:5173 and
    // 127.0.0.1:5173 (backend/app/core/config.py). Silently falling back to
    // 5174 when the port is taken would surface as an opaque CORS failure.
    port: 5173,
    strictPort: true,
  },
  test: {
    // The default environment. What is under test here is arithmetic over a
    // Float32Array, so a DOM would be a jsdom dependency bought for nothing —
    // the canvas component itself is not unit-tested, and would need a real
    // browser rather than a simulated one to be worth testing.
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
