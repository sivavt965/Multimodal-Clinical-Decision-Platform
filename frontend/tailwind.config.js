/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // ── Font stack ──────────────────────────────────────────────────
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'sans-serif',
        ],
      },

      // ── Elevation ───────────────────────────────────────────────────
      boxShadow: {
        'card':       '0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04)',
        'card-hover': '0 8px 20px rgba(15, 23, 42, 0.10), 0 3px 6px rgba(15, 23, 42, 0.05)',
        'inset-focus':'inset 0 0 0 2px rgba(37, 99, 235, 0.20)',
        'ring-blue':  '0 0 0 3px rgba(37, 99, 235, 0.18)',
      },

      // ── Semantic colors ─────────────────────────────────────────────
      colors: {
        brand: {
          50:  '#EFF6FF',
          100: '#DBEAFE',
          500: '#3B82F6',
          600: '#2563EB',
          700: '#1D4ED8',
          800: '#1E40AF',
        },
        ink: {
          50:  '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
        },
      },

      // ── Spacing scale extensions ────────────────────────────────────
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
      },

      // ── Border radius ───────────────────────────────────────────────
      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
      },

      // ── Keyframes ───────────────────────────────────────────────────
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%':   { opacity: '0', transform: 'translateY(14px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInDown: {
          '0%':   { opacity: '0', transform: 'translateY(-8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideInRight: {
          '0%':   { opacity: '0', transform: 'translateX(18px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        scaleIn: {
          '0%':   { opacity: '0', transform: 'scale(0.96)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0.55' },
        },
      },

      // ── Animation utilities ─────────────────────────────────────────
      animation: {
        fadeIn:       'fadeIn 0.3s ease-out forwards',
        fadeInUp:     'fadeInUp 0.35s ease-out forwards',
        fadeInDown:   'fadeInDown 0.3s ease-out forwards',
        slideInRight: 'slideInRight 0.3s ease-out forwards',
        scaleIn:      'scaleIn 0.25s ease-out forwards',
        pulseSoft:    'pulseSoft 2s ease-in-out infinite',
      },

      // ── Transition durations ────────────────────────────────────────
      transitionDuration: {
        '400': '400ms',
      },
    },
  },
  plugins: [],
}
