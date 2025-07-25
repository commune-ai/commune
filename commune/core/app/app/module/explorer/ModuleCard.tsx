'use client'

import { useState, memo, useCallback, useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { ModuleType } from '../../types/module'
import { CopyButton } from '@/app/components/CopyButton'
import Link from 'next/link'

// Helper functions
const shorten = (key: string, length = 6): string => {
  if (!key || typeof key !== 'string') return ''
  if (key.length <= length * 2 + 3) return key
  return `${key.slice(0, length)}...${key.slice(-length)}`
}

const time2str = (time: number): string => {
  const d = new Date(time * 1000)
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  
  if (diff < 60000) return 'now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`
  
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toLowerCase()
}

// Generate unique color based on module name
const getModuleColor = (name: string): string => {
  const colors = [
    '#00ff00', '#ff00ff', '#00ffff', '#ffff00', '#ff6600', 
    '#0099ff', '#ff0099', '#99ff00', '#9900ff', '#00ff99',
    '#ff9900', '#00ff66', '#6600ff', '#ff0066', '#66ff00'
  ]
  
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
  }
  
  return colors[Math.abs(hash) % colors.length]
}

// Generate cyberpunk pattern based on key
const generateCyberpunkPattern = (key: string, color: string): string => {
  if (!key) return ''
  
  // Create a deterministic pattern based on the key
  const canvas = `<svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
        <rect width="40" height="40" fill="black"/>
        <rect width="1" height="40" fill="${color}20" x="20"/>
        <rect width="40" height="1" fill="${color}20" y="20"/>
      </pattern>
      <filter id="glow">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>
    <rect width="400" height="400" fill="url(#grid)"/>`
  
  // Generate circuit-like patterns based on key hash
  let paths = ''
  let hash = 0
  for (let i = 0; i < key.length; i++) {
    hash = key.charCodeAt(i) + ((hash << 5) - hash)
  }
  
  // Create random but deterministic circuit paths
  const rand = (seed: number) => {
    const x = Math.sin(seed) * 10000
    return x - Math.floor(x)
  }
  
  for (let i = 0; i < 8; i++) {
    const x1 = Math.floor(rand(hash + i) * 400)
    const y1 = Math.floor(rand(hash + i + 1) * 400)
    const x2 = Math.floor(rand(hash + i + 2) * 400)
    const y2 = Math.floor(rand(hash + i + 3) * 400)
    
    paths += `<path d="M${x1},${y1} L${x2},${y2}" stroke="${color}" stroke-width="2" opacity="0.6" filter="url(#glow)"/>`
    
    // Add nodes
    if (i % 2 === 0) {
      paths += `<circle cx="${x1}" cy="${y1}" r="4" fill="${color}" filter="url(#glow)"/>`
      paths += `<circle cx="${x2}" cy="${y2}" r="4" fill="${color}" filter="url(#glow)"/>`
    }
  }
  
  // Add some data blocks
  for (let i = 0; i < 5; i++) {
    const x = Math.floor(rand(hash + i + 10) * 350) + 25
    const y = Math.floor(rand(hash + i + 11) * 350) + 25
    const size = 20 + Math.floor(rand(hash + i + 12) * 30)
    
    paths += `<rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${color}15" stroke="${color}" stroke-width="1" filter="url(#glow)"/>`
  }
  
  const svg = canvas + paths + '</svg>'
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

interface ModuleCardProps {
  module: ModuleType
  viewMode?: 'grid' | 'list'
}

const ModuleCard = memo(({ module, viewMode = 'grid' }: ModuleCardProps) => {
  const router = useRouter()
  const [isHovered, setIsHovered] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const moduleColor = getModuleColor(module.name)
  
  // Generate cyberpunk pattern
  const cyberpunkPattern = useMemo(() => {
    return generateCyberpunkPattern(module.key, moduleColor)
  }, [module.key, moduleColor])
  
  const handleCardClick = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsLoading(true)
    await router.push(`module/${module.name}?color=${encodeURIComponent(moduleColor)}`)
  }, [router, module.name, moduleColor])



  return (
    <div
      onClick={handleCardClick}
      className='cursor-pointer border-2 bg-black p-6 font-mono hover:shadow-2xl transition-all duration-300 relative overflow-hidden group flex flex-col h-[420px]'
      style={{ 
        borderColor: moduleColor,
        boxShadow: isHovered ? `0 0 30px ${moduleColor}80, inset 0 0 20px ${moduleColor}20` : `0 0 10px ${moduleColor}40`,
        borderWidth: isHovered ? '3px' : '2px',
        background: isHovered ? `radial-gradient(ellipse at center, ${moduleColor}10 0%, black 70%)` : 'black'
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Loading overlay */}
      {isLoading && (
        <div className='absolute inset-0 z-20 bg-black/90 flex items-center justify-center'>
          <div className='text-5xl font-bold animate-pulse text-white'>{`loading...`}</div>
        </div>
      )}
      
      {/* Main content wrapper */}
      <div className='flex-1 flex flex-col relative z-10'>
        {/* Header section with module name */}
        <div className='mb-4'>
          <h3 className='font-bold text-4xl tracking-wider lowercase transition-all duration-300 text-white mb-3' 
              style={{ 
                textShadow: isHovered ? `0 0 15px ${moduleColor}80` : `0 0 5px ${moduleColor}40`,
                letterSpacing: '0.05em'
              }}>
            {module.name}
          </h3>
          
          {/* Info section with pattern beside keys */}
          <div className='flex gap-4'>
            {/* Left side - Keys and info */}
            <div className='flex flex-col gap-2 flex-1'>
              {/* Key */}
              <div className='flex items-center gap-2'>
                <span className='text-xl lowercase font-bold text-white/60'>key</span>
                <div className='border px-3 py-1 rounded-md' style={{ borderColor: `${moduleColor}50`, backgroundColor: `${moduleColor}10` }}>
                  <code className='text-xl font-mono text-white/80'>
                    {shorten(module.key, 4)}
                  </code>
                </div>
                <div onClick={(e) => e.stopPropagation()}>
                  <CopyButton code={module.key} />
                </div>
              </div>
              
              {/* CID */}
              {module.cid && (
                <div className='flex items-center gap-2'>
                  <span className='text-xl lowercase font-bold text-white/60'>cid</span>
                  <div className='border px-3 py-1 rounded-md' style={{ borderColor: `${moduleColor}50`, backgroundColor: `${moduleColor}10` }}>
                    <code className='text-xl font-mono text-white/80'>
                      {shorten(module.cid, 4)}
                    </code>
                  </div>
                  <div onClick={(e) => e.stopPropagation()}>
                    <CopyButton code={module.cid} />
                  </div>
                </div>
              )}
              
              {/* Date */}
              <div className='flex items-center gap-2'>
                <span className='text-xl lowercase font-bold text-white/60'>time</span>
                <div className='border px-3 py-1 rounded-md' style={{ borderColor: `${moduleColor}50`, backgroundColor: `${moduleColor}10` }}>
                  <span className='text-xl font-mono text-white/80'>
                    {time2str(module.time)}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Right side - Pattern image */}
            {cyberpunkPattern && (
              <div 
                className='w-48 h-48 rounded-lg overflow-hidden border-2 opacity-70 group-hover:opacity-90 transition-opacity duration-300'
                style={{
                  borderColor: `${moduleColor}40`,
                  backgroundImage: `url(${cyberpunkPattern})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center'
                }}
              />
            )}
          </div>
        </div>

        {/* Description section */}
        {module.desc && (
          <div className='flex-1 flex flex-col justify-center mb-4'>
            <p className='text-2xl leading-relaxed line-clamp-2 text-white/70'>
              {module.desc}
            </p>
          </div>
        )}

        {/* Bottom section with tags */}
        <div className='mt-auto'>
          {/* Tags at the bottom */}
          <div className='flex flex-wrap gap-2 min-h-[32px]'>
            {module.tags && module.tags.length > 0 ? (
              <>
                {module.tags.slice(0, 8).map((tag, i) => (
                  <span
                    key={i}
                    className='text-lg border px-3 py-1 lowercase tracking-wide transition-all duration-200 hover:scale-110 text-white'
                    style={{ 
                      borderColor: `${moduleColor}50`,
                      backgroundColor: `${moduleColor}15`,
                      boxShadow: isHovered ? `0 0 10px ${moduleColor}40` : 'none'
                    }}
                  >
                    {tag}
                  </span>
                ))}
                {module.tags.length > 8 && (
                  <span className='text-lg px-3 py-1 text-white/60'>+{module.tags.length - 8}</span>
                )}
              </>
            ) : (
              <span className='text-lg text-white/40 italic'>no tags</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
})

ModuleCard.displayName = 'ModuleCard'

export default ModuleCard