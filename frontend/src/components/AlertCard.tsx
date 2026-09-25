import { formatNumber, severityTheme } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

/** Hero card: the headline prediction, its AQI and the plain-language alert. */
export default function AlertCard({ data }: { data: ForecastResponse }) {
  const theme = severityTheme(data.alert.severity)
  const { forecast, alert, aqi } = data

  return (
    <section
      className={`card card-pad animate-fade-up border-2 ${theme.card}`}
      aria-label="Air exposure alert"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="stat-label">Alert for the next hour</p>
          <h2 className="mt-1 text-2xl font-bold text-white sm:text-3xl">
            {alert.headline}
          </h2>
        </div>
        <span className={`chip ${theme.badge}`}>
          <span aria-hidden="true">●</span>
          {alert.category}
        </span>
      </div>

      <div className="mt-5 grid gap-5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <div>
          <p className="text-slate-200">{alert.advice}</p>
          <p className="mt-3 rounded-xl border border-white/10 bg-ink-900/50 p-3 text-sm text-slate-300">
            <span className="font-semibold text-slate-200">Sensitive groups: </span>
            {alert.sensitive_group_advice}
          </p>
        </div>

        <div className="text-left sm:text-right">
          <p className="stat-label">Predicted PM2.5</p>
          <p className={`text-4xl font-bold tabular-nums ${theme.accent}`}>
            {formatNumber(forecast.predicted_pm25)}
          </p>
          <p className="text-sm text-slate-400">µg/m³ · {alert.horizon}</p>
          <p className="mt-2 text-xs text-slate-500">
            US EPA AQI {aqi.aqi} · {formatNumber(aqi.who_ratio, 2)}× WHO 24h guideline
          </p>
        </div>
      </div>
    </section>
  )
}