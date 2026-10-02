import { exposureTheme, formatDateTime, formatSignedPercent } from '../lib/theme'
import type { SpikeOut } from '../lib/types'

const SEVERITY_BADGE: Record<string, string> = {
  none: 'bg-white/10 text-slate-300',
  minor: 'bg-yellow-500/20 text-yellow-200',
  moderate: 'bg-orange-500/20 text-orange-200',
  severe: 'bg-red-500/20 text-red-200',
}

/**
 * Pollution spike card.
 *
 * The confidence is a real, model-derived value: either an empirically
 * calibrated historical frequency or a transparent rule-based estimate. The
 * basis is always stated. Contributing signals are described as associated
 * signals, never as proven causes.
 */
export default function SpikeCard({ spike }: { spike: SpikeOut | null }) {
  if (!spike) return null

  if (!spike.spike_detected) {
    return (
      <section className="card card-pad animate-fade-up" aria-label="Pollution spike">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-semibold text-white">Pollution spike</h3>
          <span className="chip bg-emerald-500/20 text-emerald-300">No spike expected</span>
        </div>
        <p className="mt-2 text-sm text-slate-300">{spike.message}</p>
        <p className="mt-2 text-xs text-slate-500">
          Checked against the forecast timeline; confidence basis: {spike.confidence_basis}.
        </p>
      </section>
    )
  }

  const theme = exposureTheme(
    spike.severity === 'severe' ? 'high' : spike.severity === 'moderate' ? 'moderate' : 'low',
  )

  return (
    <section
      className="card card-pad animate-fade-up border-2 border-orange-400/40 bg-orange-400/5"
      aria-label="Pollution spike"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="stat-label">Pollution spike expected</p>
          <h3 className="mt-1 text-xl font-bold text-white">
            {spike.kind === 'sustained' ? 'Sustained rise' : 'Sharp rise'} in PM2.5
          </h3>
        </div>
        <span className={`chip ${SEVERITY_BADGE[spike.severity] ?? SEVERITY_BADGE.none}`}>
          {spike.severity} severity
        </span>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <div>
          <p className="stat-label">Expected time</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {spike.expected_time ? formatDateTime(spike.expected_time) : '—'}
          </p>
        </div>
        <div>
          <p className="stat-label">Expected change</p>
          <p className={`mt-1 text-lg font-semibold ${theme.badge.split(' ')[1] ?? 'text-white'}`}>
            {formatSignedPercent(spike.expected_change_percent)}
          </p>
          <p className="text-xs text-slate-400">
            from {spike.baseline_pm25.toFixed(1)} to {spike.peak_pm25?.toFixed(1) ?? '—'} µg/m³
          </p>
        </div>
        <div>
          <p className="stat-label">Confidence</p>
          <p className="mt-1 text-lg font-semibold text-white">
            {(spike.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-slate-400">
            basis: {spike.confidence_basis.replace(/_/g, ' ')}
          </p>
        </div>
      </div>

      {spike.associated_signals.length > 0 && (
        <div className="mt-4 rounded-xl border border-white/10 bg-ink-900/50 p-3">
          <p className="stat-label">Possible contributing signals</p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {spike.associated_signals.map((signal) => (
              <li
                key={signal.label}
                className="chip bg-white/10 text-slate-200"
                title={signal.detail ?? undefined}
              >
                {signal.label}
              </li>
            ))}
          </ul>
          {spike.signal_disclaimer && (
            <p className="mt-2 text-xs text-amber-200/80">{spike.signal_disclaimer}</p>
          )}
        </div>
      )}
    </section>
  )
}
