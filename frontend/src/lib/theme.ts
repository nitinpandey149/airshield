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

/** US EPA AQI bands, used for the scale strip. */
export const AQI_BANDS = [
  { max: 50, label: 'Good', hex: '#22c55e' },
  { max: 100, label: 'Moderate', hex: '#eab308' },
  { max: 150, label: 'Sensitive', hex: '#f97316' },
  { max: 200, label: 'Unhealthy', hex: '#ef4444' },
  { max: 300, label: 'Very unhealthy', hex: '#a855f7' },
  { max: 500, label: 'Hazardous', hex: '#7f1d1d' },
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
