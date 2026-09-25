/**
 * Typed API client for the AirShield backend.
 *
 * Errors carry the backend's own message so the UI can show the real reason a
 * request failed instead of a generic "something went wrong".
 */
import type {
  ForecastResponse,
  HealthResponse,
  LocationOut,
  ModelInfoResponse,
} from './types'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { Accept: 'application/json' },
      ...init,
    })
  } catch (cause) {
    throw new ApiError(
      'Could not reach the AirShield API. Is the backend running on port 8000?',
      0,
    )
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      const body = await response.json()
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      // Response body was not JSON; keep the status-based message.
    }
    throw new ApiError(detail, response.status)
  }

  return (await response.json()) as T
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  locations: () => request<LocationOut[]>('/api/locations'),
  modelInfo: () => request<ModelInfoResponse>('/api/model'),
  forecast: (slug: string) => request<ForecastResponse>(`/api/forecast/${slug}`),
}
