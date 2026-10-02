import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { NATIONAL_AQI_BANDS, formatHour } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

// India CPCB PM2.5 thresholds (24h, ug/m3) for the six National AQI bands.
const PM25_BANDS = [
  { y: 30, hex: NATIONAL_AQI_BANDS[0].hex },
  { y: 60, hex: NATIONAL_AQI_BANDS[1].hex },
  { y: 90, hex: NATIONAL_AQI_BANDS[2].hex },
  { y: 120, hex: NATIONAL_AQI_BANDS[3].hex },
  { y: 250, hex: NATIONAL_AQI_BANDS[4].hex },
]

/**
 * Pollution timeline: past measured PM2.5 followed by the multi-horizon
 * forecast. Measured and predicted series are visually distinct (solid sky
 * blue versus dashed), and AQI category bands are shaded behind them.
 */
export default function PollutionTimeline({ data }: { data: ForecastResponse }) {
  const series = data.history.map((point, index) => {
    const next = data.history[index + 1]
    const isBoundary = !point.predicted && next?.predicted
    return {
      label: formatHour(point.time),
      time: point.time,
      measured: point.predicted ? null : point.pm2_5,
      predicted: point.predicted ? point.pm2_5 : isBoundary ? point.pm2_5 : null,
    }
  })

  const boundaryLabel = formatHour(data.forecast.base_time)
  const horizons = data.forecasts.map((f) => f.horizon_hours).join(', ')

  return (
    <section className="card card-pad animate-fade-up" aria-label="Pollution timeline">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-white">Pollution timeline</h3>
          <p className="text-xs text-slate-400">
            Measured history and the model forecast{horizons ? ` (${horizons}h horizons)` : ''}
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded bg-sky-400" aria-hidden="true" />
            Measured
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="h-0.5 w-4 rounded border-t-2 border-dashed border-violet-400"
              aria-hidden="true"
            />
            Predicted
          </span>
        </div>
      </div>

      <div className="mt-4 h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
            {PM25_BANDS.map((band, index) => (
              <ReferenceArea
                key={band.y}
                y1={band.y}
                y2={PM25_BANDS[index + 1]?.y ?? band.y * 1.6}
                fill={band.hex}
                fillOpacity={0.05}
                ifOverflow="extendDomain"
              />
            ))}

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
            <XAxis
              dataKey="label"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              interval={Math.max(0, Math.floor(series.length / 8) - 1)}
              minTickGap={16}
            />
            <YAxis
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{
                value: 'µg/m³',
                angle: -90,
                position: 'insideLeft',
                fill: '#64748b',
                fontSize: 11,
                offset: 22,
              }}
            />
            <Tooltip
              contentStyle={{
                background: '#111827',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 12,
                fontSize: 12,
              }}
              labelStyle={{ color: '#e2e8f0' }}
              formatter={(value: number, name: string) => [
                `${value.toFixed(1)} µg/m³`,
                name === 'measured' ? 'Measured' : 'Predicted',
              ]}
            />

            <ReferenceLine
              x={boundaryLabel}
              stroke="#64748b"
              strokeDasharray="4 4"
              label={{ value: 'now', fill: '#64748b', fontSize: 10, position: 'top' }}
            />

            <Line
              type="monotone"
              dataKey="measured"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={false}
              connectNulls={false}
              name="measured"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#a78bfa"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={{ r: 3, fill: '#a78bfa', strokeWidth: 0 }}
              connectNulls
              name="predicted"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-xs text-slate-500">
        Shaded bands are India CPCB National AQI PM2.5 categories. The dashed line is the model's
        forecast, not a measurement.
      </p>
    </section>
  )
}
