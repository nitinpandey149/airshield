/** Presentation helpers: severity theming, AQI scale and formatting. */
import type { Severity } from './types'

interface SeverityTheme {
  label: string
  /** Tailwind classes for the alert card. */
  card: string
  badge: string
  accent: string
  /** Hex used for charts, which cannot read Tailwind classes. */
  hex: string
}

export const SEVERITY_THEME: Record<Severity, SeverityTheme> = {
  good: {
    label: 'Good',
    card: 'border-shield-good/40 bg-shield-good/10',
    badge: 'bg-shield-good text-ink-900',
    accent: 'text-shield-good',
    hex: '#22c55e',
  },
  moderate: {
    label: 'Moderate',
    card: 'border-shield-moderate/40 bg-shield-moderate/10',
    badge: 'bg-shield-moderate text-ink-900',
    accent: 'text-shield-moderate',
    hex: '#eab308',
  },
  elevated: {
    label: 'Elevated',
    card: 'border-shield-elevated/45 bg-shield-elevated/10',
    badge: 'bg-shield-elevated text-white',
    accent: 'text-shield-elevated',
    hex: '#f97316',
  },
  high: {
    label: 'High',
    card: 'border-shield-high/45 bg-shield-high/10',
    badge: 'bg-shield-high text-white',
    accent: 'text-shield-high',
    hex: '#ef4444',
  },
  very_high: {
    label: 'Very high',
    card: 'border-shield-very_high/45 bg-shield-very_high/10',
    badge: 'bg-shield-very_high text-white',
    accent: 'text-shield-very_high',
    hex: '#a855f7',
  },
  hazardous: {
    label: 'Hazardous',
    card: 'border-red-900/60 bg-red-950/40',
    badge: 'bg-shield-hazardous text-white',
    accent: 'text-red-300',
    hex: '#7f1d1d',
  },
}

/** US EPA AQI bands, kept as a reference alongside the Indian scale. */
export const AQI_BANDS = [
  { max: 50, label: 'Good', hex: '#22c55e' },
  { max: 100, label: 'Moderate', hex: '#eab308' },
  { max: 150, label: 'Sensitive', hex: '#f97316' },
  { max: 200, label: 'Unhealthy', hex: '#ef4444' },
  { max: 300, label: 'Very unhealthy', hex: '#a855f7' },
  { max: 500, label: 'Hazardous', hex: '#7f1d1d' },
] as const

/**
 * India CPCB National AQI bands with their PM2.5 thresholds (24h, µg/m³).
 *
 * Charts draw these as horizontal bands because the app forecasts PM2.5, and
 * the Indian PM2.5 thresholds differ substantially from the US EPA ones - using
 * the EPA lines on an Indian scale would mislabel the air.
 */
export const NATIONAL_AQI_BANDS = [
  { max: 50, pm25: 30, label: 'Good', hex: '#22c55e' },
  { max: 100, pm25: 60, label: 'Satisfactory', hex: '#eab308' },
  { max: 200, pm25: 90, label: 'Moderate', hex: '#f97316' },
  { max: 300, pm25: 120, label: 'Poor', hex: '#ef4444' },
  { max: 400, pm25: 250, label: 'Very poor', hex: '#a855f7' },
  { max: 500, pm25: 500, label: 'Severe', hex: '#7f1d1d' },
] as const

export function severityTheme(severity: Severity): SeverityTheme {
  return SEVERITY_THEME[severity] ?? SEVERITY_THEME.good
}

/** Percentage along a 0-500 AQI scale, clamped for display. */
export function aqiPercent(aqi: number): number {
  return Math.min(100, Math.max(0, (aqi / 500) * 100))
}

export function formatNumber(value: number, digits = 1): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

/** Format an ISO timestamp as a local hour label, e.g. "14:00". */
export function formatHour(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** A time window as "6:00 AM – 6:45 AM". */
export function formatTimeRange(startIso: string, endIso: string): string {
  return `${formatHour(startIso)} – ${formatHour(endIso)}`
}

/** Relative day prefix, so a window tomorrow is not mistaken for today. */
export function formatDayLabel(iso: string): string {
  const date = new Date(iso)
  const now = new Date()
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const days = Math.round((startOfDay(date) - startOfDay(now)) / 86_400_000)
  if (days === 0) return 'today'
  if (days === 1) return 'tomorrow'
  return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })
}

interface ExposureTheme {
  label: string
  badge: string
  hex: string
}

/** Colours for the relative exposure levels produced by the backend. */
export const EXPOSURE_THEME: Record<string, ExposureTheme> = {
  low: { label: 'Low', badge: 'bg-emerald-500/20 text-emerald-300', hex: '#22c55e' },
  moderate: { label: 'Moderate', badge: 'bg-yellow-500/20 text-yellow-200', hex: '#eab308' },
  high: { label: 'High', badge: 'bg-orange-500/20 text-orange-200', hex: '#f97316' },
  very_high: { label: 'Very high', badge: 'bg-red-500/20 text-red-200', hex: '#ef4444' },
}

export function exposureTheme(level: string): ExposureTheme {
  return EXPOSURE_THEME[level] ?? { label: level, badge: 'bg-white/10 text-slate-200', hex: '#94a3b8' }
}

/** Format a signed percentage, e.g. "+35%". */
export function formatSignedPercent(value: number): string {
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(0)}%`
}
