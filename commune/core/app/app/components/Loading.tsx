import React from 'react'

export function Loading() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black">
      {/* subtle matrix scanlines */}
      <div className="pointer-events-none absolute inset-0 opacity-20 mix-blend-screen [background-image:repeating-linear-gradient(0deg,rgba(0,255,0,0.08)_0px,rgba(0,255,0,0.08)_1px,transparent_1px,transparent_3px)]" />

      <div className="relative w-64 h-64">
        {/* SNAKE FRAME */}
        <svg
          className="absolute inset-0 drop-shadow-[0_0_8px_#00ff00]"
          viewBox="0 0 120 120"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Glow filter for extra neon */}
          <defs>
            <filter id="g" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="1.2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Track (pixel grid-ish via strokeDasharray) */}
          <rect
            x="10"
            y="10"
            width="100"
            height="100"
            rx="6"
            ry="6"
            fill="none"
            stroke="#053c05"
            strokeWidth="3"
            strokeDasharray="2 2"
            shapeRendering="crispEdges"
          />

          {/* Snake trail (faint afterglow) */}
          <rect
            x="10"
            y="10"
            width="100"
            height="100"
            rx="6"
            ry="6"
            fill="none"
            stroke="#00ff00"
            strokeWidth="6"
            strokeLinecap="square"
            filter="url(#g)"
            pathLength={100}
            /* trail a bit longer but faint */
            style={{
              strokeDasharray: '16 84', // 16% snake, 84% gap
              animation: 'dash-move 2.4s linear infinite',
              opacity: 0.18,
            }}
          />

          {/* Main snake segment */}
          <rect
            x="10"
            y="10"
            width="100"
            height="100"
            rx="6"
            ry="6"
            fill="none"
            stroke="#00ff00"
            strokeWidth="6"
            strokeLinecap="square"
            pathLength={100}
            filter="url(#g)"
            style={{
              strokeDasharray: '12 88', // visible length of the snake
              animation: 'dash-move 2.4s linear infinite',
            }}
          />

          {/* Snake “head” highlight (shorter, brighter segment slightly offset) */}
          <rect
            x="10"
            y="10"
            width="100"
            height="100"
            rx="6"
            ry="6"
            fill="none"
            stroke="#b6ffb6"
            strokeWidth="6"
            strokeLinecap="square"
            pathLength={100}
            style={{
              strokeDasharray: '4 96',
              animation: 'dash-move 2.4s linear infinite',
              animationDelay: '-.12s',
              opacity: 0.9,
            }}
          />

          {/* Corner pixels that “blink” as the head passes (timed offsets) */}
          {[
            { x: 10, y: 10, d: '0s' },     // top-left
            { x: 105, y: 10, d: '-.6s' },  // top-right
            { x: 105, y: 105, d: '-1.2s' },// bottom-right
            { x: 10, y: 105, d: '-1.8s' }, // bottom-left
          ].map((c, i) => (
            <rect
              key={i}
              x={c.x - 2}
              y={c.y - 2}
              width="6"
              height="6"
              fill="#00ff00"
              style={{ animation: 'blink 2.4s linear infinite', animationDelay: c.d, opacity: 0.25 }}
              shapeRendering="crispEdges"
            />
          ))}
        </svg>

        {/* Retro terminal text */}
        <div className="absolute bottom-3 left-0 right-0 text-center">
          <span
            className="text-green-500 font-mono text-xs"
            style={{ letterSpacing: '2px', textShadow: '0 0 6px #00ff00' }}
          >
            ROUTING SNAKE…
          </span>
        </div>
      </div>

      {/* Animations */}
      <style jsx>{`
        @keyframes dash-move {
          0%   { stroke-dashoffset: 0; }
          100% { stroke-dashoffset: -100; }
        }
        @keyframes blink {
          0%, 80%, 100% { opacity: 0.18; }
          40% { opacity: 0.9; }
        }
      `}</style>
    </div>
  )
}
