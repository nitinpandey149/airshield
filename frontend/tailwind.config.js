/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Air-themed palette used across the UI.
        ink: {
          900: '#0a0f1a',
          800: '#111827',
          700: '#1b2436',
          600: '#27334a',
        },
        shield: {
          good: '#22c55e',
          moderate: '#eab308',
          elevated: '#f97316',
          high: '#ef4444',
          very_high: '#a855f7',
          hazardous: '#7f1d1d',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.35s ease-out both',
      },
    },
  },
  plugins: [],
}
