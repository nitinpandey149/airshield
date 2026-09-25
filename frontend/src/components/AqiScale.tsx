import { aqiPercent } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

/** Horizontal EPA AQI scale with the current value marked. */
export default function AqiScale({ data }: { data: ForecastResponse }) {
  const { aqi } = data
  const percent = aqiPercent(aqi.aqi)

  return (
    <section className="card card-pad animate-fade-up" aria-label="AQI scale">
      <div className="flex items-baseline justify-between">
        <h3 className="font-semibold text-white">Air Quality Index</h3>
        <p className="text-sm text-slate-400">
          AQI <span className="font-semibold text-white">{aqi.aqi}</span> · {aqi.category}
        </p>
      </div>

      <div className="mt-4">
        <div className="relative h-3 w-full overflow-hidden rounded-full">
          <div className="flex h-full w-full">
            <div className="h-full flex-1 bg-shield-good" />
            <div className="h-full flex-1 bg-shield-moderate" />
            <div className="h-full flex-1 bg-shield-elevated" />
            <div className="h-full flex-1 bg-shield-high" />
            <div className="h-full flex-1 bg-shield-very_high" />
            <div className="h-full flex-1 bg-shield-hazardous" />
          </div>
          {/* Marker */}
          <div
            className="absolute top-1/2 h-6 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_0_2px_rgba(10,15,26,0.9)] transition-all duration-500"
            style={{ left: `${percent}%` }}
            role="img"
            aria-label={`AQI ${aqi.aqi}, ${aqi.category}`}
          />
        </div>
        <div className="mt-2 flex justify-between text-[10px] uppercase tracking-wide text-slate-500">
          <span>0</span>
          <span>50</span>
          <span>100</span>
          <span>150</span>
          <span>200</span>
          <span>300+</span>
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <dt className="stat-label">Category band</dt>
          <dd className="mt-1 font-medium text-slate-200">{aqi.band}</dd>
        </div>
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <dt className="stat-label">vs WHO 24h</dt>
          <dd className="mt-1 font-medium text-slate-200">
            {aqi.who_ratio.toFixed(2)}× guideline
          </dd>
        </div>
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <dt className="stat-label">Horizon</dt>
          <dd className="mt-1 font-medium text-slate-200">
            {data.forecast.horizon_hours}h ahead
          </dd>
        </div>
      </dl>
    </section>
  )
}