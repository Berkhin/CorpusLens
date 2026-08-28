/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Origin of the FastAPI backend. Optional — defaults to
   * `http://localhost:8000`, the origin the API's CORS policy expects to be
   * called from Vite's dev server on :5173.
   */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
