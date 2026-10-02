import type { AwsArchitecture, AwsStatusResponse } from '../lib/types'

/**
 * AWS architecture and deployment status.
 *
 * The status reflects what is actually configured in the running process. It
 * never claims a service is deployed when it is not, and local inference works
 * fully without AWS.
 */
export default function AwsPanel({
  status,
  architecture,
}: {
  status: AwsStatusResponse | null
  architecture: AwsArchitecture | null
}) {
  if (!status || !architecture) return null

  return (
    <section className="card card-pad animate-fade-up" aria-label="AWS architecture">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-white">AWS architecture</h3>
          <p className="text-xs text-slate-400">
            Region {status.region} · inference backend {status.inference_backend}
          </p>
        </div>
        <span
          className={`chip ${
            status.sagemaker_active
              ? 'bg-emerald-500/20 text-emerald-300'
              : 'bg-white/10 text-slate-300'
          }`}
        >
          {status.sagemaker_active ? 'SageMaker active' : 'SageMaker not active'}
        </span>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <p className="stat-label">Forecast pipeline</p>
          <ol className="mt-2 space-y-1 text-sm text-slate-300">
            {architecture.pipeline.map((step, index) => (
              <li key={step} className="flex items-center gap-2">
                <span className="text-xs text-slate-500">{index + 1}.</span>
                {step}
              </li>
            ))}
          </ol>
          <p className="stat-label mt-3">Notification pipeline</p>
          <ol className="mt-2 space-y-1 text-sm text-slate-300">
            {architecture.notification_pipeline.map((step, index) => (
              <li key={step} className="flex items-center gap-2">
                <span className="text-xs text-slate-500">{index + 1}.</span>
                {step}
              </li>
            ))}
          </ol>
        </div>

        <div>
          <p className="stat-label">Service status</p>
          <ul className="mt-2 space-y-1.5 text-sm">
            {status.components.map((component) => (
              <li key={component.key} className="flex items-start justify-between gap-3">
                <span className="text-slate-200">
                  {component.name}
                  <span className="block text-xs text-slate-500">{component.purpose}</span>
                </span>
                <span
                  className={`chip shrink-0 ${
                    component.configured
                      ? 'bg-emerald-500/20 text-emerald-300'
                      : 'bg-white/10 text-slate-400'
                  }`}
                  title={component.detail}
                >
                  {component.configured ? 'configured' : 'off'}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-white/10 bg-ink-900/40 p-3 text-xs text-slate-400">
        <p>
          <span className="text-slate-300">CloudWatch:</span> {architecture.observability}
        </p>
        <p className="mt-1">
          <span className="text-slate-300">ML service:</span> {architecture.ml_service}
        </p>
        <p className="mt-1">
          <span className="text-slate-300">Infrastructure as code:</span>{' '}
          <span className="font-mono">{architecture.infrastructure_as_code}</span>
        </p>
        <p className="mt-1 text-amber-200/80">{status.note}</p>
      </div>
    </section>
  )
}
