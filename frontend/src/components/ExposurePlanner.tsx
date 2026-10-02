import { useState } from 'react'
import { api, ApiError } from '../lib/api'
import {
  exposureTheme,
  formatDayLabel,
  formatNumber,
  formatTimeRange,
} from '../lib/theme'
import type { ActivityOption, ExposurePlanResponse, WindowOut } from '../lib/types'

const DURATIONS = [15, 30, 45, 60]
const TIME_PRESETS: { key: string; label: string; start: number; end: number }[] = [
  { key: 'any', label: 'Any time', start: 0, end: 24 },
  { key: 'morning', label: 'Morning', start: 5, end: 12 },
  { key: 'afternoon', label: 'Afternoon', start: 12, end: 17 },
  { key: 'evening', label: 'Evening', start: 17, end: 23 },
]

function WindowRow({
  title,
  window: win,
  tone,
}: {
  title: string
  window: WindowOut
  tone: 'best' | 'alt' | 'worst'
}) {
  const theme = exposureTheme(win.exposure.level)
  const border =
    tone === 'best'
      ? 'border-emerald-400/40 bg-emerald-400/5'
      : tone === 'worst'
        ? 'border-red-400/30 bg-red-400/5'
        : 'border-white/10 bg-ink-900/40'

  return (
    <div className={`rounded-xl border p-3 ${border}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</p>
        <span className={`chip ${theme.badge}`}>{win.exposure.level_label}</span>
      </div>
      <p className="mt-1 text-lg font-semibold text-white">
        {formatTimeRange(win.start, win.end)}
      </p>
      <p className="text-xs text-slate-400">{formatDayLabel(win.start)}</p>
      <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
        <div>
          <dt className="text-slate-500">Exposure</dt>
          <dd className="text-slate-200">{formatNumber(win.exposure.score, 1)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Mean PM2.5</dt>
          <dd className="text-slate-200">{formatNumber(win.exposure.mean_pm25)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Peak</dt>
          <dd className="text-slate-200">{formatNumber(win.exposure.peak_pm25)}</dd>
        </div>
      </dl>
      {win.relative_reduction_percent !== null && win.relative_reduction_percent > 0 && (
        <p className="mt-2 text-xs font-medium text-emerald-300">
          {formatNumber(win.relative_reduction_percent, 0)}% lower predicted exposure than the
          highest-exposure window
        </p>
      )}
    </div>
  )
}

/**
 * Exposure planner.
 *
 * The user picks an activity, a duration and a preferred time band; the backend
 * scores every candidate window over the whole duration and ranks them. The
 * result describes lower predicted exposure, never safety.
 */
export default function ExposurePlanner({
  slug,
  activities,
}: {
  slug: string
  activities: ActivityOption[]
}) {
  const [activity, setActivity] = useState('walking')
  const [duration, setDuration] = useState(45)
  const [timePreset, setTimePreset] = useState('any')
  const [plan, setPlan] = useState<ExposurePlanResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    setLoading(true)
    setError(null)
    try {
      const preset = TIME_PRESETS.find((p) => p.key === timePreset) ?? TIME_PRESETS[0]
      const now = new Date()
      const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
      const start = preset.key === 'any' ? undefined : new Date(startOfToday.getTime() + preset.start * 3600_000).toISOString()
      const end = preset.key === 'any' ? undefined : new Date(startOfToday.getTime() + preset.end * 3600_000).toISOString()
      const result = await api.plan(slug, {
        activity,
        duration_minutes: duration,
        start_time: start,
        end_time: end,
      })
      setPlan(result)
    } catch (cause) {
      setPlan(null)
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="card card-pad animate-fade-up" aria-label="Exposure planner">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-white">Exposure planner</h3>
          <p className="text-xs text-slate-400">
            Find the lowest-exposure window for your activity over its whole duration.
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <label className="text-sm">
          <span className="stat-label">Activity</span>
          <select
            className="mt-1 w-full rounded-lg border border-white/10 bg-ink-900 px-3 py-2 text-slate-100"
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
          >
            {activities.map((option) => (
              <option key={option.activity} value={option.activity}>
                {option.label} (×{option.intensity})
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm">
          <span className="stat-label">Duration</span>
          <select
            className="mt-1 w-full rounded-lg border border-white/10 bg-ink-900 px-3 py-2 text-slate-100"
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
          >
            {DURATIONS.map((minutes) => (
              <option key={minutes} value={minutes}>
                {minutes} minutes
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm">
          <span className="stat-label">Preferred time</span>
          <select
            className="mt-1 w-full rounded-lg border border-white/10 bg-ink-900 px-3 py-2 text-slate-100"
            value={timePreset}
            onChange={(e) => setTimePreset(e.target.value)}
          >
            {TIME_PRESETS.map((preset) => (
              <option key={preset.key} value={preset.key}>
                {preset.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        type="button"
        onClick={submit}
        disabled={loading || activities.length === 0}
        className="mt-4 rounded-lg bg-sky-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:opacity-50"
      >
        {loading ? 'Scoring windows…' : 'Find my best window'}
      </button>

      {error && (
        <p role="alert" className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-100">
          {error}
        </p>
      )}

      {plan && (
        <div className="mt-5 space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <WindowRow title="Best window" window={plan.best_window} tone="best" />
            {plan.alternative_window && (
              <WindowRow title="Alternative window" window={plan.alternative_window} tone="alt" />
            )}
          </div>

          {plan.highest_exposure_window && (
            <WindowRow title="Avoid" window={plan.highest_exposure_window} tone="worst" />
          )}

          {plan.relative_reduction_percent !== null && (
            <p className="rounded-lg border border-emerald-400/30 bg-emerald-400/5 p-3 text-sm text-emerald-100">
              Approximately {formatNumber(plan.relative_reduction_percent, 0)}% lower predicted
              exposure than the worst available window for {plan.activity_label.toLowerCase()} over{' '}
              {plan.duration_minutes} minutes.
            </p>
          )}

          <p className="text-xs text-slate-500">{plan.reason}</p>
          <p className="text-xs text-amber-200/80">{plan.note}</p>
          <p className="text-xs text-slate-500">{plan.exposure_note}</p>
        </div>
      )}
    </section>
  )
}
