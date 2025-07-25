'use client'

import { useEffect, useState, useMemo, useCallback } from 'react'
import { Footer } from '@/app/components'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import ModuleCard from '@/app/module/explorer/ModuleCard'
import { CreateModule } from '@/app/module/explorer/ModuleCreate'
import { ModuleType } from '@/app/types/module'
import { AdvancedSearch, SearchFilters } from '@/app/module/explorer/search'
import { filterModules } from '@/app/module/explorer/search'
import { ChevronLeft, ChevronRight, Filter, Search, X, Menu } from 'lucide-react'

// Vibey ASCII Loading Component with fixed height
function VibeLoader() {
  const [frame, setFrame] = useState(0)
  const [dots, setDots] = useState('')
  const [glitch, setGlitch] = useState(false)
  const [matrixRain, setMatrixRain] = useState<Array<{char: string, y: number, speed: number}>>([])
  
  const asciiFrames = [
    `
     ╔══════════════════════════════════════╗
     ║  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  ║
     ║  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█  ║
     ║  █░░╔═╗░╔═╗░╔╗░░╔╗░╔╗░░╔╗░╔═╗░░█  ║
     ║  █░░║░╚╗║░║░║║░░║║░║║░░║║░║░╚╗░█  ║
     ║  █░░║░░║║░║░║║░░║║░║║░░║║░║░░║░█  ║
     ║  █░░║░╔╝║░║░║║░░║║░║║░░║║░║░╔╝░█  ║
     ║  █░░╚═╝░╚═╝░╚╝░░╚╝░╚╝░░╚╝░╚═╝░░█  ║
     ║  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█  ║
     ║  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀  ║
     ╚══════════════════════════════════════╝`,
    `
     ╔══════════════════════════════════════╗
     ║  ┌─────────────────────────────────┐  ║
     ║  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ║
     ║  │▓╔═══╗╔═══╗╔╗──╔╗╔╗──╔╗╔═══╗▓▓▓│  ║
     ║  │▓║╔═╗║║╔═╗║║║──║║║║──║║║╔═╗║▓▓▓│  ║
     ║  │▓║║─╚╝║║─║║║╚╗╔╝║║╚╗╔╝║║║─║║▓▓▓│  ║
     ║  │▓║║─╔╗║║─║║║╔╗╔╗║║╔╗╔╗║║║─║║▓▓▓│  ║
     ║  │▓║╚═╝║║╚═╝║║║╚╝║║║║╚╝║║║╚═╝║▓▓▓│  ║
     ║  │▓╚═══╝╚═══╝╚╝──╚╝╚╝──╚╝╚═══╝▓▓▓│  ║
     ║  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ║
     ║  └─────────────────────────────────┘  ║
     ╚══════════════════════════════════════╝`,
    `
     ╔══════════════════════════════════════╗
     ║  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ║
     ║  ░█▀▀░█▀█░█▄█░█▄█░█░█░█▀█░█▀▀░░░░░  ║
     ║  ░█░░░█░█░█░█░█░█░█░█░█░█░█▀▀░░░░░  ║
     ║  ░▀▀▀░▀▀▀░▀░▀░▀░▀░▀▀▀░▀░▀░▀▀▀░░░░░  ║
     ║  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ║
     ║  ░▒▓█▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█▓▒░░░  ║
     ║  ░▒▓█░░░░░░░░░░░░░░░░░░░░░░░█▓▒░░░  ║
     ║  ░▒▓█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█▓▒░░░  ║
     ║  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ║
     ╚══════════════════════════════════════╝`
  ]
  
  const loadingMessages = [
    "INITIALIZING QUANTUM TUNNELS",
    "SYNCHRONIZING NEURAL MATRICES",
    "CALIBRATING DATA STREAMS",
    "ESTABLISHING HYPERLINKS",
    "LOADING MODULE MANIFESTS",
    "PARSING COSMIC METADATA"
  ]
  
  const glitchChars = '!@#$%^&*()_+-=[]{}|;:,.<>?░▒▓█▀▄■□'
  
  // Initialize matrix rain
  useEffect(() => {
    const chars = Array.from({ length: 15 }, (_, i) => ({
      char: String.fromCharCode(65 + Math.floor(Math.random() * 26)),
      y: Math.random() * -20,
      speed: 0.5 + Math.random() * 1.5
    }))
    setMatrixRain(chars)
  }, [])
  
  // Update matrix rain positions
  useEffect(() => {
    const interval = setInterval(() => {
      setMatrixRain(prev => prev.map(drop => ({
        ...drop,
        y: drop.y > 20 ? -20 : drop.y + drop.speed,
        char: Math.random() > 0.95 ? String.fromCharCode(65 + Math.floor(Math.random() * 26)) : drop.char
      })))
    }, 50)
    return () => clearInterval(interval)
  }, [])
  
  useEffect(() => {
    const frameInterval = setInterval(() => {
      setFrame(prev => (prev + 1) % asciiFrames.length)
    }, 500)
    return () => clearInterval(frameInterval)
  }, [])
  
  useEffect(() => {
    const dotsInterval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.')
    }, 400)
    return () => clearInterval(dotsInterval)
  }, [])
  
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      setGlitch(true)
      setTimeout(() => setGlitch(false), 100)
    }, 3000)
    return () => clearInterval(glitchInterval)
  }, [])
  
  const currentMessage = loadingMessages[Math.floor(frame / asciiFrames.length * loadingMessages.length)]
  
  const glitchText = (text: string) => {
    if (!glitch) return text
    return text.split('').map(char => 
      Math.random() > 0.7 ? glitchChars[Math.floor(Math.random() * glitchChars.length)] : char
    ).join('')
  }
  
  return (
    <div className='flex items-center justify-center' style={{ minHeight: '400px', height: '400px' }}>
      <div className='relative'>
        {/* ASCII Art Frame */}
        <pre className='text-green-500 font-mono text-xs sm:text-sm animate-pulse'>
          {asciiFrames[frame]}
        </pre>
        
        {/* Glitch overlay */}
        {glitch && (
          <div className='absolute inset-0 flex items-center justify-center'>
            <pre className='text-green-400 font-mono text-xs sm:text-sm opacity-50'>
              {glitchText(asciiFrames[(frame + 1) % asciiFrames.length])}
            </pre>
          </div>
        )}
      </div>
      
      {/* Side content */}
      <div className='ml-8 relative' style={{ width: '300px' }}>
        {/* Loading message */}
        <div className='text-center mb-4'>
          <div className='text-green-500 font-mono text-sm mb-2'>
            [{glitchText(currentMessage)}]
          </div>
          <div className='text-green-400 font-mono text-lg'>
            LOADING{dots}
          </div>
        </div>
        
        {/* Progress bar */}
        <div className='w-full mb-4'>
          <div className='h-2 bg-black border border-green-500 rounded-full overflow-hidden'>
            <div 
              className='h-full bg-gradient-to-r from-green-500 via-green-400 to-green-500 animate-pulse'
              style={{
                width: `${((frame + 1) / asciiFrames.length) * 100}%`,
                transition: 'width 0.5s ease-out'
              }}
            />
          </div>
        </div>
        
        {/* Matrix rain effect */}
        <div className='relative h-20 overflow-hidden'>
          {matrixRain.map((drop, i) => (
            <div
              key={i}
              className='absolute font-mono text-xs text-green-400'
              style={{
                left: `${i * 20}px`,
                top: `${drop.y}px`,
                opacity: Math.max(0, 1 - (drop.y / 20))
              }}
            >
              {drop.char}
            </div>
          ))}
        </div>
        
        {/* Random tech stats */}
        <div className='mt-4 space-y-1 text-xs font-mono'>
          <div className='text-green-500/70'>
            CPU: {Math.floor(20 + Math.random() * 60)}%
          </div>
          <div className='text-green-500/70'>
            MEM: {Math.floor(30 + Math.random() * 50)}%
          </div>
          <div className='text-green-500/70'>
            NET: {Math.floor(100 + Math.random() * 900)}KB/s
          </div>
        </div>
      </div>
    </div>
  )
}

interface ModulesState {
  modules: ModuleType[]
  allModules: ModuleType[]  // Store all modules for client-side filtering
  totalModules: number
  loading: boolean
  error: string | null
}

export default function Modules() {
  const client = new Client()
  const [page, setPage] = useState(1)
  const [pageSize] = useState(9)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false) // Default to closed
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({
    searchTerm: '',
    includeTags: [],
    excludeTags: [],
    includeTerms: [],
    excludeTerms: []
  })
  const [state, setState] = useState<ModulesState>({
    modules: [],
    allModules: [],
    totalModules: 0,
    loading: false,
    error: null
  })

  // Apply client-side filtering
  const filteredModules = useMemo(() => {
    if (!searchFilters.searchTerm && 
        searchFilters.includeTerms.length === 0 &&
        searchFilters.excludeTerms.length === 0) {
      return state.modules
    }
    return filterModules(state.modules, searchFilters)
  }, [state.modules, searchFilters])

  // Pagination calculations
  const totalFilteredPages = Math.ceil(filteredModules.length / pageSize)
  const paginatedModules = filteredModules.slice((page - 1) * pageSize, page * pageSize)
  const hasNextPage = page < totalFilteredPages
  const hasPrevPage = page > 1

  const fetchModules = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      // Fetch all modules for client-side filtering
      const [modulesData, totalCount] = await Promise.all([
        client.call('modules', { page_size: 1000 }), // Get more modules for better filtering
        client.call('n', {})
      ])
      
      if (!Array.isArray(modulesData)) {
        throw new Error('Invalid response format')
      }

      setState({
        modules: modulesData,
        allModules: modulesData,
        totalModules: totalCount,
        loading: false,
        error: null
      })
    } catch (err: any) {
      setState({
        modules: [],
        allModules: [],
        totalModules: 0,
        loading: false,
        error: err.message || 'Failed to fetch modules'
      })
    }
  }

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalFilteredPages) return
    setPage(newPage)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleSearch = useCallback((filters: SearchFilters) => {
    setSearchFilters(filters)
    setPage(1) // Reset to first page on search
  }, [])

  const handleCreateSuccess = () => {
    fetchModules()
    setShowCreateForm(false)
    setState(prev => ({ ...prev, error: null }))
  }

  useEffect(() => {
    fetchModules()
  }, [])

  const PaginationControls = () => (
    <nav className='flex items-center gap-2 font-mono text-sm' aria-label='Pagination'>
      <button
        onClick={() => handlePageChange(page - 1)}
        disabled={!hasPrevPage || state.loading}
        className='px-3 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-none'
        aria-label='Previous page'
      >
        [PREV]
      </button>
      
      <span className='px-3 text-green-500'>
        PAGE {page}/{totalFilteredPages || 1}
      </span>
      
      <button
        onClick={() => handlePageChange(page + 1)}
        disabled={!hasNextPage || state.loading}
        className='px-3 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-none'
        aria-label='Next page'
      >
        [NEXT]
      </button>
      
      <span className='ml-4 text-green-500' aria-live='polite'>
        SHOWING: {paginatedModules.length} / {filteredModules.length} FILTERED
      </span>
    </nav>
  )

  return (
    <div className='min-h-screen bg-black text-green-500 font-mono'>
      {/* Error Banner */}
      {state.error && (
        <div className='flex items-center justify-between p-4 bg-black border border-red-500 text-red-500' role='alert'>
          <span>ERROR: {state.error}</span>
          <button 
            onClick={() => setState(prev => ({ ...prev, error: null }))} 
            className='hover:bg-red-500 hover:text-black px-2 transition-none'
            aria-label='Dismiss error'
          >
            [X]
          </button>
        </div>
      )}

      {/* Create Module Modal */}
      {showCreateForm && (
        <div className='fixed inset-0 z-50 flex items-center justify-center bg-black'>
          <div className='border border-green-500 bg-black p-4'>
            <CreateModule
              onClose={() => setShowCreateForm(false)}
              onSuccess={handleCreateSuccess}
            />
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className='p-6' role='main'>
        {/* Search Section - Now at the top */}
        <div className='mb-6'>
          {/* Search Bar with Toggle */}
          <div className='mb-4'>
            <AdvancedSearch
              onSearch={handleSearch}
              availableTags={[]}
              isExpanded={showAdvancedSearch}
              onToggleExpanded={() => setShowAdvancedSearch(!showAdvancedSearch)}
            />
          </div>
          
          {/* Quick Stats when search is expanded */}
          {showAdvancedSearch && (
            <div className='bg-black/40 border border-green-500/30 rounded p-4 mb-4'>
              <h3 className='text-green-500 text-sm font-bold mb-3 uppercase'>Module Stats</h3>
              <div className='grid grid-cols-2 md:grid-cols-4 gap-4 text-xs'>
                <div>
                  <span className='text-green-500/70'>Total Modules:</span>
                  <span className='text-green-500 ml-2'>{state.totalModules}</span>
                </div>
                <div>
                  <span className='text-green-500/70'>Filtered:</span>
                  <span className='text-green-500 ml-2'>{filteredModules.length}</span>
                </div>
                <div>
                  <span className='text-green-500/70'>Showing:</span>
                  <span className='text-green-500 ml-2'>{paginatedModules.length}</span>
                </div>
                <div>
                  <span className='text-green-500/70'>Page:</span>
                  <span className='text-green-500 ml-2'>{page}/{totalFilteredPages || 1}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Loading State */}
        {state.loading && <VibeLoader />}
        
        {/* Empty State */}
        {!state.loading && paginatedModules.length === 0 && (
          <div className='py-8 text-center text-green-500' role='status'>
            {searchFilters.searchTerm || searchFilters.includeTerms.length > 0 || searchFilters.excludeTerms.length > 0
              ? 'NO MODULES MATCH YOUR FILTERS.'
              : 'NO MODULES AVAILABLE.'}
          </div>
        )}

        {/* Modules Grid */}
        <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' role='list'>
          {paginatedModules.map((m) => (
            <div key={m.key} role='listitem'>
              <ModuleCard module={m} />
            </div>
          ))}
        </div>
      </main>

      {/* Pagination Controls Bottom */}
      {!state.loading && paginatedModules.length > 0 && (
        <div className='flex justify-center p-6 border-t border-green-500'>
          <PaginationControls />
        </div>
      )}

      <Footer />
    </div>
  )
}