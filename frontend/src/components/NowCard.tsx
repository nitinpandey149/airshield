import { formatNumber, severityTheme } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

/**
 * The answer to "should I go outside right now?".
 *
 * It combines the current measured hour with the next-hour prediction and the
 * spike outlook, then states a single plain-language verdict. It never claims
 * medical safety - it reports relative exposure.
 */
export default function NowCard({ data }: { data: ForecastResponse }) {
  const { current, forecast, aqi, alert, spike, national_aqi: national } = data
  const theme = severityTheme(alert.severity)

  const rising = forecast.predicted_pm25 > current.pm2_5 * 1.1
  const falling = forecast.predicted_pm25 < current.pm2_5 * 0.9
  const trend = rising ? 'rising' : falling ? 'falling' : 'steady'
  const trendArrow = rising ? '↑' : falling ? '↓' : '→'

  const verdict = spike?.spike_detected
    ? 'A pollution spike is expected soon — consider going out earlier or later.'
    : theme.label === 'Good' || theme.label === 'Moderate'
      ? 'Conditions are relatively favourable right now.'
      : 'Current air is elevated — keep outdoor time short where you can.'

  return (
    <section
      className={`card card-pad animate-fade-up border-2 ${theme.card}`}
      aria-label="Current conditions and outlook"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="stat-label">Should I go outside right now?</p>
          <h2 className="mt-1 text-xl font-bold text-white sm:text-2xl">{verdict}</h2>
        </div>
        <span className={`chip ${theme.badge}`}>
          <span aria-hidden="true">●</span>
          {national ? `National AQI ${national.category}` : current.category}
        </span>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <p className="stat-label">Current PM2.5</p>
          <p className="stat-value">{formatNumber(current.pm2_5)}</p>
          <p className="text-xs text-slate-400">
            µg/m³ · measured {new Date(current.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
          {national ? (
            <p className="mt-1 text-xs text-slate-500">
              India National AQI {national.aqi} · {national.category}
              {national.dominant_pollutant !== 'pm2_5' &&
                ` (set by ${national.dominant_pollutant.toUpperCase()})`}
            </p>
          ) : (
            <p className="mt-1 text-xs text-slate-500">
              {formatNumber(current.who_ratio, 2)}× WHO guideline
            </p>
          )}
        </div>

        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <p className="stat-label">Next hour</p>
          <p className={`text-2xl font-semibold tabular-nums ${theme.accent}`}>
            {formatNumber(forecast.predicted_pm25)}
          </p>
          <p className="text-xs text-slate-400">
            µg/m³ · {forecast.horizon_hours}h forecast {trendArrow} {trend}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            National AQI {alert.aqi} · {alert.category}
          </p>
        </div>

        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <p className="stat-label">Spike outlook</p>
          <p className="text-2xl font-semibold tabular-nums text-white">
            {spike?.spike_detected ? `${(spike.confidence * 100).toFixed(0)}%` : 'None'}
          </p>
          <p className="text-xs text-slate-400">
            {spike?.spike_detected
              ? `${spike.severity} · ${spike.expected_change_percent > 0 ? '+' : ''}${spike.expected_change_percent.toFixed(0)}%`
              : 'no significant rise predicted'}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            confidence basis: {spike?.confidence_basis?.replace(/_/g, ' ') ?? 'n/a'}
          </p>
        </div>
      </div>

      <p className="mt-4 text-sm text-slate-300">{alert.advice}</p>
      <p className="mt-2 text-xs text-slate-500">
        Alert basis: {alert.basis}. US EPA AQI {aqi.aqi} for the same value (different scale).
      </p>
    </section>
  )
}
