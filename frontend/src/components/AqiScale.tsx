import { aqiPercent } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

/** India CPCB National AQI bands, used for the scale strip. */
const NATIONAL_BANDS = [
  { max: 50, label: 'Good', className: 'bg-shield-good' },
  { max: 100, label: 'Satisfactory', className: 'bg-shield-moderate' },
  { max: 200, label: 'Moderate', className: 'bg-shield-elevated' },
  { max: 300, label: 'Poor', className: 'bg-shield-high' },
  { max: 400, label: 'Very poor', className: 'bg-shield-very_high' },
  { max: 500, label: 'Severe', className: 'bg-shield-hazardous' },
] as const

/**
 * India CPCB National AQI scale with the current value marked.
 *
 * The National AQI is the worst sub-index across the pollutants the source
 * provides. When a pollutant the standard covers is unavailable (CO, SO2, NH3),
 * the card says so rather than implying full coverage.
 */
export default function AqiScale({ data }: { data: ForecastResponse }) {
  const national = data.national_aqi
  const { aqi } = data
  const percent = aqiPercent(national ? national.aqi : aqi.aqi)

  return (
    <section className="card card-pad animate-fade-up" aria-label="AQI scale">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-semibold text-white">
          {national ? 'India National AQI' : 'US EPA Air Quality Index'}
        </h3>
        <p className="text-sm text-slate-400">
          {national ? (
            <>
              AQI <span className="font-semibold text-white">{national.aqi}</span> ·{' '}
              {national.category}
            </>
          ) : (
            <>
              AQI <span className="font-semibold text-white">{aqi.aqi}</span> · {aqi.category}
            </>
          )}
        </p>
      </div>

      <div className="mt-4">
        <div className="relative h-3 w-full overflow-hidden rounded-full">
          <div className="flex h-full w-full">
            {NATIONAL_BANDS.map((band) => (
              <div key={band.label} className={`h-full flex-1 ${band.className}`} />
            ))}
          </div>
          <div
            className="absolute top-1/2 h-6 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_0_2px_rgba(10,15,26,0.9)] transition-all duration-500"
            style={{ left: `${percent}%` }}
            role="img"
            aria-label={
              national
                ? `National AQI ${national.aqi}, ${national.category}`
                : `AQI ${aqi.aqi}, ${aqi.category}`
            }
          />
        </div>
        <div className="mt-2 flex justify-between text-[10px] uppercase tracking-wide text-slate-500">
          <span>0</span>
          <span>50</span>
          <span>100</span>
          <span>200</span>
          <span>300</span>
          <span>400</span>
          <span>500</span>
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <dt className="stat-label">Dominant pollutant</dt>
          <dd className="mt-1 font-medium text-slate-200">
            {national
              ? `${national.dominant_pollutant.toUpperCase()} (${national.band})`
              : aqi.band}
          </dd>
        </div>
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <dt className="stat-label">vs WHO 24h</dt>
          <dd className="mt-1 font-medium text-slate-200">
            {(national ? national.who_ratio : aqi.who_ratio).toFixed(2)}× guideline
          </dd>
        </div>
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <dt className="stat-label">Horizon</dt>
          <dd className="mt-1 font-medium text-slate-200">
            {data.forecast.horizon_hours}h ahead
          </dd>
        </div>
      </dl>

      {national && national.sub_indices.length > 0 && (
        <div className="mt-3">
          <p className="stat-label">Pollutant sub-indices</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {national.sub_indices.map((s) => (
              <span key={s.pollutant} className="chip bg-white/10 text-slate-200">
                {s.pollutant.toUpperCase()} {s.sub_index} · {s.category}
              </span>
            ))}
          </div>
        </div>
      )}

      {national && (
        <p className="mt-3 text-xs text-slate-500">
          {national.standard}. {national.is_partial
            ? `This source does not provide ${national.missing_pollutants.join(', ')}, so the value may be lower than the official National AQI.`
            : 'All National AQI pollutants were available.'}{' '}
          {national.health_guidance}
        </p>
      )}
      {!national && (
        <p className="mt-3 text-xs text-slate-500">
          US EPA scale, shown alongside the Indian National AQI in the headline card.
        </p>
      )}
    </section>
  )
}