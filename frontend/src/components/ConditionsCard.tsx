import { formatNumber } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

function Metric({ label, value, unit }: { label: string; value: number | null; unit: string }) {
  return (
    <div>
      <dt className="stat-label">{label}</dt>
      <dd className="mt-0.5 text-sm text-slate-200">
        {value === null ? '—' : `${formatNumber(value)} ${unit}`}
      </dd>
    </div>
  )
}

/** Supporting environmental context for the current hour, from real data. */
export default function ConditionsCard({ data }: { data: ForecastResponse }) {
  const { current } = data
  const w = current.weather

  return (
    <section className="card card-pad animate-fade-up" aria-label="Environmental conditions">
      <h3 className="font-semibold text-white">Conditions right now</h3>
      <p className="text-xs text-slate-400">
        Measured at {new Date(current.time).toLocaleString()}
      </p>

      <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="PM10" value={current.pm10} unit="µg/m³" />
        <Metric label="NO₂" value={current.nitrogen_dioxide} unit="µg/m³" />
        <Metric label="O₃" value={current.ozone} unit="µg/m³" />
        <Metric label="Wind" value={w.wind_speed_10m} unit="km/h" />
        <Metric label="Temperature" value={w.temperature_2m} unit="°C" />
        <Metric label="Humidity" value={w.relative_humidity_2m} unit="%" />
        <Metric label="Pressure" value={w.surface_pressure} unit="hPa" />
        <Metric label="Precipitation" value={w.precipitation} unit="mm" />
      </dl>

      <p className="mt-4 text-xs text-slate-500">
        These are measured values from the upstream source, shown alongside the
        forecast for context. They are not used to claim a cause for any change in
        PM2.5.
      </p>
    </section>
  )
}
