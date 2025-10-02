import type { Config } from 'tailwindcss';
export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: { neon: { bg:'#0a0b0f', panel:'#0b0e17', border:'#1e293b', text:'#e2e8f0', cyan:'#22d3ee', mag:'#ff2ea6', lime:'#b4ff39' } },
      boxShadow: { glow:'0 0 24px rgba(34,211,238,.35)', mg:'0 0 24px rgba(255,46,166,.35)' },
      backgroundImage: { scan:'repeating-linear-gradient(0deg,rgba(255,255,255,.03) 0,rgba(255,255,255,.03) 1px,transparent 3px,transparent 4px)' }
    }
  },
  plugins: []
} satisfies Config;
