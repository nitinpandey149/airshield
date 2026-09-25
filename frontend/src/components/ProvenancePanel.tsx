import { formatDateTime } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

/**
 * Provenance panel.
 *
 * Always visible, never collapsed: it states whether the numbers on screen come
 * from live measurements or the bundled historical sample, and repeats the
 * upstream licence.
 */
export default function ProvenancePanel({ data }: { data: ForecastResponse }) {
  const { source, notice } = data
  const isDemo = source.mode === 'demo'

  return (
    <section className="card card-pad animate-fade-up" aria-label="Data provenance">
      {notice && (
        <div
          className="mb-4 rounded-xl border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-200"
          role="status"
        >
          <p className="font-semibold">⚠ Demo data in use</p>
          <p className="mt-1 text-amber-100/90">{notice}</p>
        </div>
      )}

      <h3 className="font-semibold text-white">Where these numbers come from</h3>

      <dl className="mt-3 space-y-3 text-sm">
        <div className="flex items-start justify-between gap-3">
          <dt className="stat-label">Data source</dt>
          <dd className="text-right text-slate-200">
            {source.name}
            <span
              className={`ml-2 chip ${isDemo ? 'bg-amber-400/20 text-amber-200' : 'bg-emerald-500/20 text-emerald-300'}`}
            >
              {isDemo ? 'DEMO' : 'LIVE'}
            </span>
          </dd>
        </div>

        <div className="flex items-start justify-between gap-3">
          <dt className="stat-label">Licence</dt>
          <dd className="max-w-[60%] text-right text-slate-300">{source.licence}</dd>
        </div>

        <div className="flex items-start justify-between gap-3">
          <dt className="stat-label">Upstream</dt>
          <dd className="max-w-[60%] truncate text-right">
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer noopener"
              className="text-sky-400 hover:text-sky-300"
            >
              {source.url}
            </a>
          </dd>
        </div>

        <div className="flex items-start justify-between gap-3">
          <dt className="stat-label">Retrieved</dt>
          <dd className="text-right text-slate-300">
            {source.fetched_at ? formatDateTime(source.fetched_at) : '—'}
          </dd>
        </div>

        <div className="flex items-start justify-between gap-3">
          <dt className="stat-label">Prediction basis</dt>
          <dd className="text-right text-slate-300">
            {formatDateTime(data.forecast.base_time)} → {formatDateTime(data.forecast.target_time)}
          </dd>
        </div>

        <div className="flex items-start justify-between gap-3">
          <dt className="stat-label">Model</dt>
          <dd className="text-right font-mono text-xs text-slate-300">
            {data.forecast.model_version}
            <span className="ml-2 chip bg-white/10 text-slate-300">
              {data.forecast.backend === 'aws' ? 'SageMaker AI' : 'local'}
            </span>
          </dd>
        </div>
      </dl>

      <p className="mt-4 text-xs text-slate-500">
        {notice
          ? 'The measurement history above is a real historical sample from the source listed, not current conditions.'
          : 'The measurement history above is real observed data from the source listed.'}{' '}
        The prediction is produced by the model named here and is an estimate, not a
        measurement.
      </p>
    </section>
  )
}