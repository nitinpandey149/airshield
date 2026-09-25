import { useState } from 'react'
import { formatDateTime, formatNumber } from '../lib/theme'
import type { ModelInfoResponse } from '../lib/types'

/**
 * Model card.
 *
 * Shows the real held-out metrics recorded during training, next to the
 * persistence baseline they must beat. Nothing here is decorative: every number
 * comes from the artifact produced by `make train`.
 */
export default function ModelCard({ info }: { info: ModelInfoResponse | null }) {
  const [open, setOpen] = useState(false)

  if (!info) return null

  if (!info.available) {
    return (
      <section className="card card-pad" aria-label="Model information">
        <h3 className="font-semibold text-white">Model</h3>
        <p className="mt-2 text-sm text-amber-300">
          {info.note ?? 'Model information is unavailable.'}
        </p>
      </section>
    )
  }

  const { metrics, baseline_metrics: baseline, top_features: features } = info
  const hasMetrics = Object.keys(metrics ?? {}).length > 0

  const improvement =
    hasMetrics && baseline?.rmse
      ? ((baseline.rmse - metrics.rmse) / baseline.rmse) * 100
      : null

  return (
    <section className="card card-pad animate-fade-up" aria-label="Model information">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-white">Model</h3>
          <p className="text-xs text-slate-400">
            XGBoost regression · 1-hour-ahead PM2.5 ·{' '}
            {info.inference_backend === 'aws' ? 'Amazon SageMaker AI' : 'local artifact'}
          </p>
        </div>
        <span className="chip bg-white/10 text-slate-200">
          {info.model_version ?? 'unknown version'}
        </span>
      </div>

      {!hasMetrics ? (
        <p className="mt-3 text-sm text-slate-300">{info.note}</p>
      ) : (
        <>
          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
              <p className="stat-label">Held-out performance (most recent 20% of time)</p>
              <p className="mt-1 text-sm text-slate-200">
                RMSE <span className="font-semibold text-white">{formatNumber(metrics.rmse, 2)}</span>{' '}
                µg/m³ · MAE{' '}
                <span className="font-semibold text-white">{formatNumber(metrics.mae, 2)}</span> ·
                R² <span className="font-semibold text-white">{formatNumber(metrics.r2, 3)}</span>
              </p>
              <p className="mt-1 text-xs text-slate-500">
                over {formatNumber(metrics.n ?? 0, 0)} unseen hours
              </p>
            </div>

            <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
              <p className="stat-label">Persistence baseline (next hour = this hour)</p>
              <p className="mt-1 text-sm text-slate-200">
                RMSE{' '}
                <span className="font-semibold text-white">
                  {formatNumber(baseline.rmse, 2)}
                </span>{' '}
                µg/m³ · R²{' '}
                <span className="font-semibold text-white">
                  {formatNumber(baseline.r2, 3)}
                </span>
              </p>
              {improvement !== null && (
                <p
                  className={`mt-1 text-xs ${improvement > 0 ? 'text-emerald-400' : 'text-amber-300'}`}
                >
                  Model reduces RMSE by {formatNumber(improvement, 1)}% versus persistence
                </p>
              )}
            </div>
          </div>

          <dl className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
            <div>
              <dt className="stat-label">Trained</dt>
              <dd className="mt-0.5 text-slate-300">
                {info.trained_at ? formatDateTime(info.trained_at) : '—'}
              </dd>
            </div>
            <div>
              <dt className="stat-label">Rows</dt>
              <dd className="mt-0.5 text-slate-300">
                {formatNumber(info.train_rows ?? 0, 0)} train /{' '}
                {formatNumber(info.test_rows ?? 0, 0)} test
              </dd>
            </div>
            <div>
              <dt className="stat-label">Features</dt>
              <dd className="mt-0.5 text-slate-300">{info.feature_count ?? '—'}</dd>
            </div>
            <div>
              <dt className="stat-label">Data</dt>
              <dd className="mt-0.5 text-slate-300">
                {info.locations.length} locations
              </dd>
            </div>
          </dl>

          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="mt-4 text-xs font-medium text-sky-400 hover:text-sky-300"
            aria-expanded={open}
          >
            {open ? 'Hide' : 'Show'} training details
          </button>

          {open && (
            <div className="mt-3 space-y-3 text-xs text-slate-400">
              <p>
                <span className="text-slate-300">Training window:</span>{' '}
                {info.train_window.start} → {info.train_window.end}
                <br />
                <span className="text-slate-300">Test window:</span>{' '}
                {info.train_window.test_start} → {info.train_window.test_end}
                <br />
                <span className="text-slate-300">Time-ordered split:</span> the test set is
                the most recent slice of the timeline, never a random sample.
              </p>

              {features.length > 0 && (
                <div>
                  <p className="mb-1 text-slate-300">Most informative features (gain share)</p>
                  <ul className="space-y-1">
                    {features.slice(0, 8).map((item) => (
                      <li key={item.feature} className="flex items-center gap-2">
                        <span className="w-40 shrink-0 truncate font-mono text-[11px]">
                          {item.feature}
                        </span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
                          <span
                            className="block h-full rounded-full bg-sky-400"
                            style={{ width: `${Math.min(100, item.gain_share * 250)}%` }}
                          />
                        </span>
                        <span className="w-12 shrink-0 text-right tabular-nums">
                          {(item.gain_share * 100).toFixed(1)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <p>
                <span className="text-slate-300">Libraries:</span>{' '}
                {Object.entries(info.library_versions ?? {})
                  .map(([name, version]) => `${name} ${version}`)
                  .join(', ')}
              </p>
            </div>
          )}
        </>
      )}
    </section>
  )
}