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
  standard: string
}

export interface SubIndexOut {
  pollutant: string
  concentration: number
  sub_index: number
  category: string
  band: string
}

export interface NationalAqiOut {
  aqi: number
  category: string
  dominant_pollutant: string
  band: string
  who_ratio: number
  sub_indices: SubIndexOut[]
  health_guidance: string
  missing_pollutants: string[]
  is_partial: boolean
  standard: string
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
  basis: string
  health_guidance: string
}

export interface HistoryPoint {
  time: string
  pm2_5: number | null
  predicted: boolean
}

export interface WeatherOut {
  temperature_2m: number | null
  relative_humidity_2m: number | null
  wind_speed_10m: number | null
  wind_direction_10m: number | null
  surface_pressure: number | null
  precipitation: number | null
}

export interface CurrentOut {
  time: string
  pm2_5: number
  pm10: number | null
  nitrogen_dioxide: number | null
  ozone: number | null
  aqi: number
  category: string
  band: string
  who_ratio: number
  weather: WeatherOut
}

export interface HorizonForecast {
  horizon_hours: number
  predicted_pm25: number
  base_time: string
  target_time: string
  model_version: string
  backend: InferenceBackend
  aqi: number
  category: string
  band: string
  who_ratio: number
  model_metrics: Record<string, number>
}

export interface AssociatedSignal {
  label: string
  direction?: string
  detail?: string
}

export interface SpikeOut {
  spike_detected: boolean
  kind: string
  severity: string
  expected_time: string | null
  expected_change_percent: number
  peak_pm25: number | null
  baseline_pm25: number
  horizon_hours: number
  confidence: number
  confidence_basis: string
  associated_signals: AssociatedSignal[]
  signal_disclaimer?: string | null
  message: string
}

export interface ForecastResponse {
  location: LocationOut
  generated_at: string
  source: SourceInfo
  notice: string | null
  current: CurrentOut
  forecast: ForecastOut
  forecasts: HorizonForecast[]
  unavailable_horizons: string[]
  aqi: AqiOut
  national_aqi: NationalAqiOut | null
  alert: AlertOut
  spike: SpikeOut | null
  history: HistoryPoint[]
}

// -------------------------------------------------------------- planning
export interface ActivityOption {
  activity: string
  label: string
  intensity: number
}

export interface ExposureEstimate {
  score: number
  level: string
  level_label: string
  mean_pm25: number
  peak_pm25: number
  duration_minutes: number
  activity: string
  activity_label: string
  activity_intensity: number
  location_factor: number
  micro_scale: number
  basis: string
}

export interface WindowOut {
  start: string
  end: string
  exposure: ExposureEstimate
  relative_reduction_percent: number | null
}

export interface TimelinePoint {
  time: string
  pm2_5: number
  horizon_hours: number
}

export interface ExposurePlanResponse {
  location: string
  location_slug: string
  activity: string
  activity_label: string
  duration_minutes: number
  basis: string
  generated_at: string
  base_time: string
  source: SourceInfo
  notice: string | null
  best_window: WindowOut
  alternative_window: WindowOut | null
  highest_exposure_window: WindowOut | null
  relative_reduction_percent: number | null
  reason: string
  note: string
  exposure_note: string
  candidates: WindowOut[]
  timeline: TimelinePoint[]
}

// ---------------------------------------------------------------- routes
export interface RouteOut {
  route_id: string
  label: string
  distance_km: number
  duration_minutes: number
  average_pm25: number
  peak_pm25: number
  exposure_score: number
  exposure_level: string
  activity: string
  activity_label: string
  relative_reduction_percent: number | null
  provider: string
  geometry: number[][]
}

export interface RouteComparisonResponse {
  mode: string
  activity: string
  provider: string | null
  provider_profile: string | null
  generated_at: string
  routes: RouteOut[]
  recommended_route_id: string
  relative_reduction_percent: number | null
  recommendation: string
  note: string
}

// ------------------------------------------------------------------- aws
export interface AwsComponent {
  key: string
  name: string
  purpose: string
  configured: boolean
  detail: string
}

export interface AwsStatusResponse {
  region: string
  inference_backend: string
  sagemaker_active: boolean
  components: AwsComponent[]
  configured_components: string[]
  credentials: { verified: boolean; detail: string; account?: string; arn?: string }
  note: string
}

export interface AwsArchitecture {
  pipeline: string[]
  notification_pipeline: string[]
  observability: string
  region: string
  ml_service: string
  infrastructure_as_code: string
  note: string
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
  horizon_hours?: number | null
  feature_count?: number | null
  train_rows?: number | null
  test_rows?: number | null
  metrics: Record<string, number>
  baseline_metrics: Record<string, number>
  moving_average_metrics: Record<string, number>
  train_window: Record<string, string>
  locations: string[]
  data_source: Record<string, unknown>
  hyperparameters: Record<string, unknown>
  library_versions: Record<string, string>
  top_features: { feature: string; gain_share: number }[]
  note?: string | null
}

export interface HorizonCard {
  horizon_hours: number
  model_version: string | null
  trained_at: string | null
  metrics: Record<string, number>
  baseline_metrics: Record<string, number>
  moving_average_metrics: Record<string, number>
  train_rows: number | null
  test_rows: number | null
}

export interface HorizonsResponse {
  horizons: HorizonCard[]
  available: number[]
  note: string
}

// --------------------------------------------------------------- assistant
export type AssistantMode = 'llm' | 'extractive' | 'refusal' | 'llm_error'

export interface AssistantSource {
  doc_id: string
  title: string
  source: string
  url: string
  category: string
  document_type: string
  publication_date: string | null
  licence: string
}

export interface AssistantRetrieved {
  doc_id: string
  title: string
  source: string
  url: string
  score: number
  ordinal: number
}

export interface AssistantChatResponse {
  answer: string
  sources: AssistantSource[]
  retrieved_chunks: number
  grounded: boolean
  insufficient_knowledge: boolean
  mode: AssistantMode
  llm_available: boolean
  llm_model: string
  context_used: Record<string, unknown>
  retrieved: AssistantRetrieved[]
  notice: string | null
  context_error?: string | null
}

export interface SuggestedQuestion {
  id: string
  label: string
  question: string
  needs_context: boolean
}

export interface AssistantStatusResponse {
  enabled: boolean
  ready: boolean
  index: Record<string, unknown> | null
  embedder_semantic: boolean | null
  llm_available: boolean
  llm_model: string | null
  llm_detail: string | null
  suggested_questions: SuggestedQuestion[]
  notice: string | null
  detail: string | null
}

export interface AssistantChatParams {
  message: string
  location_slug?: string
  activity?: string
  duration_minutes?: number
}
