import { useState } from 'react'
import { api, ApiError } from '../lib/api'
import { exposureTheme, formatNumber } from '../lib/theme'
import type { RouteComparisonResponse, RouteOut } from '../lib/types'

const MODES = [
  { key: 'walking', label: 'Walking' },
  { key: 'cycling', label: 'Cycling' },
  { key: 'running', label: 'Running' },
]

const PRESETS: { label: string; origin: [number, number]; dest: [number, number] }[] = [
  { label: 'Berlin centre → Kreuzberg', origin: [52.52, 13.405], dest: [52.486, 13.424] },
  { label: 'Berlin centre → Prenzlauer Berg', origin: [52.52, 13.405], dest: [52.538, 13.424] },
]

function RouteCard({ route, best }: { route: RouteOut; best: boolean }) {
  const theme = exposureTheme(route.exposure_level)
  return (
    <div
      className={`rounded-xl border p-4 ${
        best ? 'border-emerald-400/50 bg-emerald-400/5' : 'border-white/10 bg-ink-900/40'
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <p className="font-semibold text-white">{route.label}</p>
          {best && (
            <span className="chip bg-emerald-500/20 text-emerald-300">
              ★ Lower predicted exposure
            </span>
          )}
        </div>
        <span className={`chip ${theme.badge}`}>{route.exposure_level.replace('_', ' ')}</span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div>
          <dt className="stat-label">Distance</dt>
          <dd className="text-slate-200">{formatNumber(route.distance_km, 2)} km</dd>
        </div>
        <div>
          <dt className="stat-label">Time</dt>
          <dd className="text-slate-200">{formatNumber(route.duration_minutes, 0)} min</dd>
        </div>
        <div>
          <dt className="stat-label">Avg PM2.5</dt>
          <dd className="text-slate-200">{formatNumber(route.average_pm25)} µg/m³</dd>
        </div>
        <div>
          <dt className="stat-label">Exposure score</dt>
          <dd className="font-semibold text-white">{formatNumber(route.exposure_score, 1)}</dd>
        </div>
      </dl>

      {route.relative_reduction_percent !== null && route.relative_reduction_percent > 0 && (
        <p className="mt-2 text-xs font-medium text-emerald-300">
          {formatNumber(route.relative_reduction_percent, 0)}% lower predicted exposure than the
          highest-exposure route
        </p>
      )}
      <p className="mt-1 text-xs text-slate-500">via {route.provider}</p>
    </div>
  )
}

/**
 * Route comparison.
 *
 * Route geometry comes from a real routing engine and predicted PM2.5 is
 * sampled along each route. The recommended route is described as having lower
 * predicted exposure, never as safe.
 */
export default function RouteComparison() {
  const [origin, setOrigin] = useState<[number, number]>(PRESETS[0].origin)
  const [dest, setDest] = useState<[number, number]>(PRESETS[0].dest)
  const [mode, setMode] = useState('walking')
  const [result, setResult] = useState<RouteComparisonResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    setLoading(true)
    setError(null)
    try {
      const data = await api.compareRoutes({
        origin_lat: origin[0],
        origin_lon: origin[1],
        dest_lat: dest[0],
        dest_lon: dest[1],
        mode,
      })
      setResult(data)
    } catch (cause) {
      setResult(null)
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="card card-pad animate-fade-up" aria-label="Route comparison">
      <h3 className="font-semibold text-white">Route comparison</h3>
      <p className="text-xs text-slate-400">
        Compare real route alternatives by predicted pollution exposure along each path.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <label className="text-sm sm:col-span-2">
          <span className="stat-label">Journey</span>
          <select
            className="mt-1 w-full rounded-lg border border-white/10 bg-ink-900 px-3 py-2 text-slate-100"
            onChange={(e) => {
              const preset = PRESETS[Number(e.target.value)]
              setOrigin(preset.origin)
              setDest(preset.dest)
            }}
          >
            {PRESETS.map((preset, index) => (
              <option key={preset.label} value={index}>
                {preset.label}
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm">
          <span className="stat-label">Travel mode</span>
          <select
            className="mt-1 w-full rounded-lg border border-white/10 bg-ink-900 px-3 py-2 text-slate-100"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            {MODES.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        type="button"
        onClick={submit}
        disabled={loading}
        className="mt-4 rounded-lg bg-sky-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:opacity-50"
      >
        {loading ? 'Comparing routes…' : 'Compare routes'}
      </button>

      {error && (
        <p role="alert" className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-100">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-5 space-y-3">
          {result.routes.map((route) => (
            <RouteCard
              key={route.route_id}
              route={route}
              best={route.route_id === result.recommended_route_id}
            />
          ))}
          <p className="rounded-lg border border-emerald-400/30 bg-emerald-400/5 p-3 text-sm text-emerald-100">
            {result.recommendation}
          </p>
          <p className="text-xs text-slate-500">
            Routing by {result.provider} ({result.provider_profile}) via OpenStreetMap.
          </p>
          <p className="text-xs text-amber-200/80">{result.note}</p>
        </div>
      )}
    </section>
  )
}
