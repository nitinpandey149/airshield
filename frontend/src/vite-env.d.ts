/// <reference types="vite/client" />

// Explicit typing for the environment variables this app reads.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
