import React, { useEffect, useState, useRef } from 'react'

export function Loading() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState(0)
  
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas size
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
    
    // Ising model parameters
    const gridSize = 100
    const cellSize = Math.min(canvas.width, canvas.height) / gridSize
    const temperature = 2.269 // Critical temperature
    const J = 1.0 // Coupling constant
    
    // Initialize spins randomly
    let spins: number[][] = Array(gridSize).fill(0).map(() => 
      Array(gridSize).fill(0).map(() => Math.random() > 0.5 ? 1 : -1)
    )
    
    // Color wave parameters
    let colorPhase = 0
    let waveOffset = 0
    
    // Metropolis algorithm step
    const metropolisStep = () => {
      for (let k = 0; k < gridSize * gridSize; k++) {
        const i = Math.floor(Math.random() * gridSize)
        const j = Math.floor(Math.random() * gridSize)
        
        // Calculate energy change
        const neighbors = 
          spins[(i + 1) % gridSize][j] +
          spins[(i - 1 + gridSize) % gridSize][j] +
          spins[i][(j + 1) % gridSize] +
          spins[i][(j - 1 + gridSize) % gridSize]
        
        const deltaE = 2 * J * spins[i][j] * neighbors
        
        // Accept or reject flip
        if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / temperature)) {
          spins[i][j] *= -1
        }
      }
    }
    
    // Animation loop
    const animate = () => {
      // Clear canvas with subtle fade effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // Update Ising model
      metropolisStep()
      
      // Update color phase and wave
      colorPhase += 0.02
      waveOffset += 0.01
      
      // Draw spins with color waves
      for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
          const x = (canvas.width - gridSize * cellSize) / 2 + j * cellSize
          const y = (canvas.height - gridSize * cellSize) / 2 + i * cellSize
          
          // Calculate distance from center for radial effects
          const centerX = gridSize / 2
          const centerY = gridSize / 2
          const distance = Math.sqrt((i - centerY) ** 2 + (j - centerX) ** 2)
          
          // Create mesmerizing color waves
          const hue = (colorPhase * 100 + distance * 5 + Math.sin(waveOffset + i * 0.1) * 60 + Math.cos(waveOffset + j * 0.1) * 60) % 360
          const saturation = 70 + Math.sin(colorPhase + distance * 0.1) * 30
          const lightness = spins[i][j] === 1 ? 60 + Math.sin(colorPhase * 2 + distance * 0.05) * 20 : 20
          
          // Apply color with glow effect for spin up states
          if (spins[i][j] === 1) {
            // Create glow effect
            const glowRadius = cellSize * 0.8
            const gradient = ctx.createRadialGradient(x + cellSize/2, y + cellSize/2, 0, x + cellSize/2, y + cellSize/2, glowRadius)
            gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.9)`)
            gradient.addColorStop(0.5, `hsla(${hue}, ${saturation}%, ${lightness * 0.8}%, 0.5)`)
            gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness * 0.6}%, 0)`)
            ctx.fillStyle = gradient
            ctx.fillRect(x - glowRadius/2 + cellSize/2, y - glowRadius/2 + cellSize/2, glowRadius, glowRadius)
          }
          
          // Draw the actual cell
          ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`
          ctx.fillRect(x, y, cellSize - 1, cellSize - 1)
          
          // Add binary text overlay for some cells
          if (Math.random() < 0.01) {
            ctx.font = `${cellSize * 0.8}px monospace`
            ctx.fillStyle = `hsla(${(hue + 180) % 360}, 100%, 70%, 0.8)`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(spins[i][j] === 1 ? '1' : '0', x + cellSize/2, y + cellSize/2)
          }
        }
      }
      
      // Draw loading text with glitch effect
      const glitchOffset = Math.random() < 0.1 ? Math.random() * 10 - 5 : 0
      ctx.save()
      ctx.translate(canvas.width / 2 + glitchOffset, canvas.height / 2)
      
      // Background for text
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(-200, -50, 400, 100)
      
      // Main text with color animation
      ctx.font = 'bold 48px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      
      // Create gradient text
      const textGradient = ctx.createLinearGradient(-200, 0, 200, 0)
      textGradient.addColorStop(0, `hsl(${colorPhase * 100}, 100%, 60%)`)
      textGradient.addColorStop(0.5, `hsl(${colorPhase * 100 + 120}, 100%, 60%)`)
      textGradient.addColorStop(1, `hsl(${colorPhase * 100 + 240}, 100%, 60%)`)
      ctx.fillStyle = textGradient
      ctx.fillText('LOADING', 0, -10)
      
      // Subtitle with wave effect
      ctx.font = '20px monospace'
      const subtitle = 'ISING MODEL SIMULATION'
      for (let i = 0; i < subtitle.length; i++) {
        const charX = (i - subtitle.length / 2) * 15
        const charY = 20 + Math.sin(colorPhase * 2 + i * 0.5) * 5
        const charHue = (colorPhase * 100 + i * 20) % 360
        ctx.fillStyle = `hsl(${charHue}, 100%, 70%)`
        ctx.fillText(subtitle[i], charX, charY)
      }
      
      ctx.restore()
      
      // Update frame counter
      setFrame(prev => prev + 1)
      
      requestAnimationFrame(animate)
    }
    
    animate()
    
    // Handle window resize
    const handleResize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    window.addEventListener('resize', handleResize)
    
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])
  
  return (
    <div className="fixed inset-0 bg-black overflow-hidden">
      <canvas 
        ref={canvasRef}
        className="absolute inset-0"
        style={{ filter: 'contrast(1.1) brightness(1.1)' }}
      />
      
      {/* Additional UI elements */}
      <div className="absolute bottom-8 left-8 font-mono text-xs space-y-1">
        <div className="text-green-400 opacity-70">
          <span className="text-yellow-400">FRAME:</span> {frame}
        </div>
        <div className="text-green-400 opacity-70">
          <span className="text-cyan-400">TEMP:</span> 2.269 (CRITICAL)
        </div>
        <div className="text-green-400 opacity-70">
          <span className="text-magenta-400">PHASE:</span> TRANSITION
        </div>
      </div>
      
      {/* Matrix rain effect overlay */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className="absolute text-green-500 font-mono text-xs"
            style={{
              left: `${i * 5}%`,
              animation: `fall ${10 + Math.random() * 10}s linear infinite`,
              animationDelay: `${Math.random() * 10}s`
            }}
          >
            {Array.from({ length: 30 }).map((_, j) => (
              <div key={j} className="opacity-0 animate-pulse" style={{ animationDelay: `${j * 0.1}s` }}>
                {Math.random() > 0.5 ? '1' : '0'}
              </div>
            ))}
          </div>
        ))}
      </div>
      
      <style jsx>{`
        @keyframes fall {
          from {
            transform: translateY(-100%);
          }
          to {
            transform: translateY(100vh);
          }
        }
      `}</style>
      
      <span className="sr-only">Loading...</span>
    </div>
  )
}