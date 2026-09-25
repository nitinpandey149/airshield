import type { LocationOut } from '../lib/types'

interface Props {
  locations: LocationOut[]
  selected: string
  onSelect: (slug: string) => void
  disabled?: boolean
}

export default function LocationSelect({
  locations,
  selected,
  onSelect,
  disabled,
}: Props) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="sr-only">Choose a location</span>
      <select
        value={selected}
        disabled={disabled}
        onChange={(event) => onSelect(event.target.value)}
        className="rounded-xl border border-white/15 bg-ink-800 px-3 py-2 text-sm font-medium text-slate-100 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-400/30 disabled:opacity-50"
      >
        {locations.map((location) => (
          <option key={location.slug} value={location.slug}>
            {location.label}
          </option>
        ))}
      </select>
    </label>
  )
}
