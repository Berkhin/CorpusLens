import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { App } from '@/App'
import { ApiError } from '@/lib/api-client'
import '@/styles/tailwind.css'

/** Retry ceiling for a localhost API — a longer backoff just delays the error. */
const MAX_RETRIES = 2

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      /**
       * The index is built offline by scripts/ingest.py and the API only ever
       * reads it, so nothing a query returns can change while the app is open.
       * Refetching would be pure waste — notably for search, where every cache
       * miss costs a CPU forward pass through CLIP's text encoder.
       */
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        // A 404 or a 422 is a verdict, not a hiccup. Everything else — a 5xx,
        // or an ApiError carrying NETWORK_ERROR_STATUS because the request
        // never reached the server — is worth another attempt.
        const isClientFault = error instanceof ApiError && error.status >= 400 && error.status < 500
        if (isClientFault) return false
        return failureCount < MAX_RETRIES
      },
    },
  },
})

const rootElement = document.getElementById('root')
if (rootElement === null) throw new Error('index.html is missing the #root mount point')

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
