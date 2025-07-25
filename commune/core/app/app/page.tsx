'use client'
import { Suspense } from 'react'
import Modules from './module/explorer/Modules'

function TerminalLoading() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="text-green-500 font-mono text-xl">
        <span>LOADING SYSTEM</span>
        <span className="cursor">_</span>
      </div>
    </div>
  )
}

export default function Home() {
  return (
    <Suspense fallback={<TerminalLoading />}>
      <Modules />
    </Suspense>
  )
}