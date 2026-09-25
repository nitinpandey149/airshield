import { useCallback, useEffect, useState } from 'react'
import AlertCard from './components/AlertCard'
import AqiScale from './components/AqiScale'
import ForecastChart from './components/ForecastChart'
import LocationSelect from './components/LocationSelect'
import ModelCard from './components/ModelCard'
import ProvenancePanel from './components/ProvenancePanel'
import { api, ApiError } from './lib/api'
import { formatDateTime } from './lib/theme'
import type {
  ForecastResponse,
  HealthResponse,
  LocationOut,
  ModelInfoResponse,
} from './lib/types'

const DEFAULT_SLUG = 'berlin'

export default function App() {
  const [locations, setLocations] = useState<LocationOut[]>([])
  const [slug, setSlug] = useState(DEFAULT_SLUG)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Initial load: discover locations, health and the model card once.
  useEffect(() => {
    let cancelled = false

    async function bootstrap() {
      try {
        const [locs, healthResponse, info] = await Promise.all([
          api.locations(),
          api.health(),
          api.modelInfo(),
        ])
        if (cancelled) return
        setLocations(locs)
        setHealth(healthResponse)
        setModelInfo(info)
        if (locs.length > 0 && !locs.some((l) => l.slug === DEFAULT_SLUG)) {
          setSlug(locs[0].slug)
        }
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : String(cause))
          setLoading(false)
        }
      }
    }

    bootstrap()
    return () => {
      cancelled = true
    }
  }, [])

  // Fetch the forecast whenever the location changes.
  const loadForecast = useCallback(async (target: string) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.forecast(target)
      setForecast(data)
    } catch (cause) {
      setForecast(null)
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadForecast(slug)
  }, [slug, loadForecast])

  const degraded = health?.status === 'degraded'

  return (
    <div className="mx-auto min-h-screen max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span aria-hidden="true" className="text-2xl">
              🛡️
            </span>
            <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              AirShield
            </h1>
          </div>
          <p className="mt-1 text-sm text-slate-400">
            Know the air before you step outside.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {health && (
            <span
              className={`chip ${
                degraded ? 'bg-amber-400/20 text-amber-200' : 'bg-emerald-500/20 text-emerald-300'
              }`}
              title={health.detail ?? undefined}
            >
              {degraded ? 'model offline' : 'model ready'}
            </span>
          )}
          <LocationSelect
            locations={locations}
            selected={slug}
            onSelect={setSlug}
            disabled={loading || locations.length === 0}
          />
        </div>
      </header>

      {/* Degraded banner explains the real reason, straight from the API. */}
      {degraded && health?.detail && (
        <div
          role="alert"
          className="mt-6 rounded-xl border border-amber-400/40 bg-amber-400/10 p-4 text-sm text-amber-100"
        >
          <p className="font-semibold">The prediction service is not ready</p>
          <p className="mt-1">{health.detail}</p>
        </div>
      )}

      <main className="mt-6 space-y-5">
        {loading && !forecast && (
          <div className="card card-pad" aria-live="polite">
            <p className="text-slate-300">Loading the latest air data…</p>
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="card card-pad border-red-500/40 bg-red-500/10"
          >
            <p className="font-semibold text-red-200">Could not load a forecast</p>
            <p className="mt-1 text-sm text-red-100/90">{error}</p>
            <button
              type="button"
              onClick={() => loadForecast(slug)}
              className="mt-3 rounded-lg border border-red-300/40 px-3 py-1.5 text-sm font-medium text-red-100 transition hover:bg-red-400/20"
            >
              Try again
            </button>
          </div>
        )}

        {forecast && (
          <>
            {loading && (
              <p className="text-xs text-slate-500" aria-live="polite">
                Refreshing…
              </p>
            )}

            <AlertCard data={forecast} />

            <div className="grid gap-5 lg:grid-cols-2">
              <ForecastChart data={forecast} />
              <AqiScale data={forecast} />
            </div>

            <div className="grid gap-5 lg:grid-cols-2">
              <ProvenancePanel data={forecast} />
              <ModelCard info={modelInfo} />
            </div>

            <footer className="pb-6 pt-2 text-xs text-slate-500">
              <p>
                {forecast.location.label} · updated{' '}
                {formatDateTime(forecast.generated_at)} · horizon{' '}
                {forecast.forecast.horizon_hours}h
              </p>
              <p className="mt-1">
                Weather and air-quality data by Open-Meteo (CC BY 4.0). Predictions by an
                XGBoost model
                {forecast.forecast.backend === 'aws'
                  ? ' served on Amazon SageMaker AI.'
                  : ' running locally from a persisted artifact.'}{' '}
                Forecasts are estimates, not medical advice.
              </p>
            </footer>
          </>
        )}
      </main>
    </div>
  )
}
