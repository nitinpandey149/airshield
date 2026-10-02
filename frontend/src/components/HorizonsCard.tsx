import { formatNumber } from '../lib/theme'
import type { HorizonsResponse } from '../lib/types'

/**
 * Multi-horizon model coverage.
 *
 * Each horizon is a separately trained model. Only horizons that actually have
 * an artifact are listed, so the card can never imply coverage that does not
 * exist. Metrics are the real held-out numbers recorded at training time.
 */
export default function HorizonsCard({ data }: { data: HorizonsResponse | null }) {
  if (!data || data.horizons.length === 0) return null

  return (
    <section className="card card-pad animate-fade-up" aria-label="Forecast horizons">
      <h3 className="font-semibold text-white">Forecast horizons</h3>
      <p className="text-xs text-slate-400">
        One XGBoost model per horizon, trained and evaluated separately.
      </p>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {data.horizons.map((card) => {
          const improvement =
            card.baseline_metrics?.rmse && card.metrics?.rmse
              ? ((card.baseline_metrics.rmse - card.metrics.rmse) / card.baseline_metrics.rmse) * 100
              : null
          return (
            <div key={card.horizon_hours} className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
              <div className="flex items-center justify-between">
                <p className="text-lg font-semibold text-white">+{card.horizon_hours}h</p>
                <span className="chip bg-white/10 text-slate-300">
                  RMSE {formatNumber(card.metrics.rmse ?? 0, 2)}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                MAE {formatNumber(card.metrics.mae ?? 0, 2)} · R²{' '}
                {formatNumber(card.metrics.r2 ?? 0, 3)}
              </p>
              {improvement !== null && (
                <p
                  className={`mt-1 text-xs ${improvement > 0 ? 'text-emerald-400' : 'text-amber-300'}`}
                >
                  {formatNumber(improvement, 1)}% better RMSE than persistence
                </p>
              )}
              <p className="mt-1 font-mono text-[10px] text-slate-500">{card.model_version}</p>
            </div>
          )
        })}
      </div>

      <p className="mt-3 text-xs text-slate-500">{data.note}</p>
    </section>
  )
}
