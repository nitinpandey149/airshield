/**
 * Typed API client for the AirShield backend.
 *
 * Errors carry the backend's own message so the UI can show the real reason a
 * request failed instead of a generic "something went wrong".
 */
import type {
  ActivityOption,
  AssistantChatParams,
  AssistantChatResponse,
  AssistantStatusResponse,
  AwsArchitecture,
  AwsStatusResponse,
  ExposurePlanResponse,
  ForecastResponse,
  HealthResponse,
  HorizonsResponse,
  LocationOut,
  ModelInfoResponse,
  RouteComparisonResponse,
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
  horizons: () => request<HorizonsResponse>('/api/horizons'),
  forecast: (slug: string) => request<ForecastResponse>(`/api/forecast/${slug}`),
  activities: () => request<ActivityOption[]>('/api/activities'),
  plan: (
    slug: string,
    params: {
      activity: string
      duration_minutes: number
      start_time?: string
      end_time?: string
    },
  ) => {
    const query = new URLSearchParams({
      activity: params.activity,
      duration_minutes: String(params.duration_minutes),
    })
    if (params.start_time) query.set('start_time', params.start_time)
    if (params.end_time) query.set('end_time', params.end_time)
    return request<ExposurePlanResponse>(`/api/plan/${slug}?${query.toString()}`)
  },
  compareRoutes: (params: {
    origin_lat: number
    origin_lon: number
    dest_lat: number
    dest_lon: number
    mode: string
  }) => {
    const query = new URLSearchParams({
      origin_lat: String(params.origin_lat),
      origin_lon: String(params.origin_lon),
      dest_lat: String(params.dest_lat),
      dest_lon: String(params.dest_lon),
      mode: params.mode,
    })
    return request<RouteComparisonResponse>(`/api/routes/compare?${query.toString()}`)
  },
  awsStatus: () => request<AwsStatusResponse>('/api/aws/status'),
  awsArchitecture: () => request<AwsArchitecture>('/api/aws/architecture'),

  assistantStatus: () => request<AssistantStatusResponse>('/api/assistant/status'),
  assistantChat: (params: AssistantChatParams) =>
    request<AssistantChatResponse>('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
}
