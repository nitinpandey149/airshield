import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AQI_BANDS, formatHour, severityTheme } from '../lib/theme'
import type { ForecastResponse } from '../lib/types'

const PM25_BANDS = [
  { y: 12, hex: AQI_BANDS[0].hex },
  { y: 35.4, hex: AQI_BANDS[1].hex },
  { y: 55.4, hex: AQI_BANDS[2].hex },
  { y: 150.4, hex: AQI_BANDS[3].hex },
  { y: 250.4, hex: AQI_BANDS[4].hex },
]

/**
 * Measured PM2.5 for the recent past, then the model's prediction.
 *
 * The predicted point is drawn with a dashed connector so the boundary between
 * what was measured and what was predicted is unmistakable.
 */
export default function ForecastChart({ data }: { data: ForecastResponse }) {
  const theme = severityTheme(data.alert.severity)

  const series = data.history.map((point, index) => {
    const isLast = index === data.history.length - 1
    return {
      time: point.time,
      label: formatHour(point.time),
      measured: point.predicted ? null : point.pm2_5,
      predicted: point.predicted || isLast ? point.pm2_5 : null,
      isPrediction: point.predicted,
    }
  })

  // "Now" marker sits between the last measurement and the prediction.
  const boundaryLabel = formatHour(data.forecast.base_time)

  return (
    <section className="card card-pad animate-fade-up" aria-label="PM2.5 trend">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold text-white">PM2.5 — recent history and next hour</h3>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded bg-sky-400" aria-hidden="true" />
            Measured
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="h-0.5 w-4 rounded border-t-2 border-dashed"
              style={{ borderColor: theme.hex }}
              aria-hidden="true"
            />
            Predicted
          </span>
        </div>
      </div>

      <div className="mt-4 h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
            <defs>
              <linearGradient id="measuredFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="predictedFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={theme.hex} stopOpacity={0.35} />
                <stop offset="100%" stopColor={theme.hex} stopOpacity={0.02} />
              </linearGradient>
            </defs>

            {/* EPA category bands, drawn as faint horizontal guides. */}
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

            <Area
              type="monotone"
              dataKey="measured"
              stroke="#38bdf8"
              strokeWidth={2}
              fill="url(#measuredFill)"
              connectNulls={false}
              dot={false}
              name="measured"
            />
            <Area
              type="monotone"
              dataKey="predicted"
              stroke={theme.hex}
              strokeWidth={2}
              strokeDasharray="5 4"
              fill="url(#predictedFill)"
              connectNulls
              dot={{ r: 3, fill: theme.hex, strokeWidth: 0 }}
              name="predicted"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-xs text-slate-500">
        The dashed segment is the model's forecast, not a measurement. Shaded bands
        mark US EPA PM2.5 categories.
      </p>
    </section>
  )
}