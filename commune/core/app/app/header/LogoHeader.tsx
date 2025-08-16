      
      
    

import { useDebounce } from '@/app/hooks/useDebounce'
import { useState } from 'react'
import { useRouter } from 'next/navigation'


export const LogoHeader = () => {
    const router = useRouter()

    return (
    <div className="relative w-8 h-10 flex-shrink-0">
    <svg viewBox="0 0 100 100" className="w-full h-full" onClick={() => {router.push('/')}} >
        <defs>
        <linearGradient id="hexGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="100%" stopColor="#34d399" />
        </linearGradient>
        </defs>
        <path
        d="M50 5 L85 25 L85 75 L50 95 L15 75 L15 25 Z"
        fill="url(#hexGradient)"
        stroke="#10b981"
        strokeWidth="2"
        className="drop-shadow-[0_0_10px_rgba(34,197,94,0.5)]"
        />

    </svg>
    </div>
    )
}
