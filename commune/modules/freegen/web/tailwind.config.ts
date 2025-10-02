import type { Config } from 'tailwindcss';
export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        neon: {
          bg: '#0a0b0f',
          grid: '#0f172a',
          panel: '#0b0e17',
          border: '#1e293b',
          text: '#e2e8f0',
          cyan: '#22d3ee',
          magenta: '#ff2ea6',
          lime: '#b4ff39'
        }
      },
      boxShadow: {
        glow: '0 0 24px rgba(34, 211, 238, 0.4)',
        mg: '0 0 24px rgba(255,46,166,.35)'
      },
      backgroundImage: {
        scan: 'repeating-linear-gradient(180deg, rgba(255,255,255,0.04) 0, rgba(255,255,255,0.04) 1px, transparent 3px, transparent 4px)',
        grid: 'radial-gradient(circle at 1px 1px, rgba(34,211,238,.15) 1px, rgba(0,0,0,0) 1px)'
      },
      backgroundSize: { grid: '24px 24px' }
    }
  },
  plugins: []
} satisfies Config;
