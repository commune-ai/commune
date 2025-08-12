import React from 'react'

export function Loading() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black">
      <div className="relative w-64 h-64">
        {/* 8-bit fractal spinner */}
        <svg className="absolute inset-0" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
          {/* Outer fractal ring */}
          <g className="animate-spin" style={{transformOrigin: '50px 50px'}}>
            <rect x="45" y="5" width="10" height="10" fill="#00ff00" opacity="0.8"/>
            <rect x="85" y="45" width="10" height="10" fill="#00ff00" opacity="0.8"/>
            <rect x="45" y="85" width="10" height="10" fill="#00ff00" opacity="0.8"/>
            <rect x="5" y="45" width="10" height="10" fill="#00ff00" opacity="0.8"/>
          </g>
          
          {/* Middle fractal squares */}
          <g className="animate-spin" style={{animationDuration: '2s', animationDirection: 'reverse', transformOrigin: '50px 50px'}}>
            <rect x="25" y="25" width="8" height="8" fill="#00ff00" opacity="0.6"/>
            <rect x="67" y="25" width="8" height="8" fill="#00ff00" opacity="0.6"/>
            <rect x="67" y="67" width="8" height="8" fill="#00ff00" opacity="0.6"/>
            <rect x="25" y="67" width="8" height="8" fill="#00ff00" opacity="0.6"/>
          </g>
          
          {/* Inner fractal pattern */}
          <g className="animate-pulse">
            <rect x="35" y="35" width="30" height="30" fill="none" stroke="#00ff00" strokeWidth="2" opacity="0.4"/>
            <rect x="40" y="40" width="20" height="20" fill="none" stroke="#00ff00" strokeWidth="2" opacity="0.5"/>
            <rect x="45" y="45" width="10" height="10" fill="#00ff00" opacity="0.7"/>
          </g>
          
          {/* Fractal corners */}
          <g className="animate-spin" style={{animationDuration: '4s', transformOrigin: '50px 50px'}}>
            <rect x="10" y="10" width="5" height="5" fill="#00ff00" opacity="0.3"/>
            <rect x="85" y="10" width="5" height="5" fill="#00ff00" opacity="0.3"/>
            <rect x="85" y="85" width="5" height="5" fill="#00ff00" opacity="0.3"/>
            <rect x="10" y="85" width="5" height="5" fill="#00ff00" opacity="0.3"/>
          </g>
          
          {/* Center pixel */}
          <rect x="48" y="48" width="4" height="4" fill="#00ff00" className="animate-pulse"/>
        </svg>
        
        {/* 8-bit style loading text */}
        <div className="absolute bottom-0 left-0 right-0 text-center">
          <span className="text-green-500 font-mono text-xs animate-pulse" style={{fontFamily: 'monospace', letterSpacing: '2px'}}>
            LOADING...
          </span>
        </div>
      </div>
    </div>
  )
}