/** Types mirroring the FastAPI response models. */

export type DataMode = 'live' | 'demo'
export type InferenceBackend = 'local' | 'aws'

export interface SourceInfo {
  name: string
  url: string
  mode: DataMode
  licence: string
  fetched_at: string | null
  is_synthetic: boolean
}

export interface LocationOut {
  slug: string
  name: string
  country: string
  latitude: number
  longitude: number
  timezone: string
  label: string
}

export interface ForecastOut {
  predicted_pm25: number
  base_time: string
  target_time: string
  horizon_hours: number
  model_version: string
  backend: InferenceBackend
  model_metrics: Record<string, number>
}

export interface AqiOut {
  aqi: number
  category: string
  band: string
  who_ratio: number
}

export type Severity =
  | 'good'
  | 'moderate'
  | 'elevated'
  | 'high'
  | 'very_high'
  | 'hazardous'

export interface AlertOut {
  severity: Severity
  headline: string
  advice: string
  sensitive_group_advice: string
  category: string
  aqi: number
  horizon: string
}

export interface HistoryPoint {
  time: string
  pm2_5: number | null
  predicted: boolean
}

export interface ForecastResponse {
  location: LocationOut
  generated_at: string
  source: SourceInfo
  notice: string | null
  forecast: ForecastOut
  aqi: AqiOut
  alert: AlertOut
  history: HistoryPoint[]
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  version: string
  inference_backend: InferenceBackend
  data_mode: string
  model_available: boolean
  model_version: string | null
  sagemaker_endpoint: string | null
  detail: string | null
}

export interface ModelInfoResponse {
  available: boolean
  inference_backend: InferenceBackend
  model_version?: string | null
  trained_at?: string | null
  feature_count?: number | null
  train_rows?: number | null
  test_rows?: number | null
  metrics: Record<string, number>
  baseline_metrics: Record<string, number>
  train_window: Record<string, string>
  locations: string[]
  data_source: Record<string, unknown>
  hyperparameters: Record<string, unknown>
  library_versions: Record<string, string>
  top_features: { feature: string; gain_share: number }[]
  note?: string | null
}
