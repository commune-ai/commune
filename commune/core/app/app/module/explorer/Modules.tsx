'use client'

import { useEffect, useState, useMemo, useCallback } from 'react'
import { Footer } from '@/app/components'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import ModuleCard from '@/app/module/explorer/ModuleCard'
import { CreateModule } from '@/app/module/explorer/ModuleCreate'
import { ModuleType } from '@/app/types/module'
import { ModSearch, SearchFilters } from '@/app/module/explorer/ModSearch'

interface ModulesState {
  modules: ModuleType[]
  n: number
  loading: boolean
  error: string | null
}

export default function Modules() {
  const client = new Client()
  const [page, setPage] = useState(1)
  const [pageSize] = useState(9)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [showModSearch, setShowModSearch] = useState(false) // Default to closed
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({
    searchTerm: '',
    includeTags: [],
    excludeTags: [],
    includeTerms: [],
    excludeTerms: []
  })
  const [state, setState] = useState<ModulesState>({
    modules: [],
    n: 0,
    loading: false,
    error: null
  })

  // Pagination calculations
  const totalPages = Math.ceil(state.n / pageSize)
  const hasNextPage = page < totalPages
  const hasPrevPage = page > 1

  const fetchModules = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      // Fetch modules for current page
      const modulesData = await client.call('modules', { page_size: pageSize, page: page , search: searchFilters.searchTerm })

      // only call 'n' once to get total count
      console.log(`Modules on page ${page}:`, modulesData)
      let n : number = state.n
      if (n === 0) {
        n = await client.call('n')
      }
      setState({
        modules: modulesData,
        n: n,
        loading: false,
        error: null
      })
    } catch (err: any) {
      setState({
        modules: [],
        n: 0,
        loading: false,
        error: err.message || 'Failed to fetch modules'
      })
    }
  }

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return
    setPage(newPage)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleSearch = useCallback((filters: SearchFilters) => {
    setSearchFilters(filters)
    setPage(1) // Reset to first page on new search
  }, [])

  const handleCreateSuccess = () => {
    fetchModules()
    setShowCreateForm(false)
    setState(prev => ({ ...prev, error: null }))
  }

  useEffect(() => {
    fetchModules()
  }, [page, searchFilters])

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
        PAGE {page}/{totalPages || 1}
      </span>
      
      <button
        onClick={() => handlePageChange(page + 1)}
        disabled={!hasNextPage || state.loading}
        className='px-3 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-none'
        aria-label='Next page'
      >
        [NEXT]
      </button>
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
        {/* Search Section with Pagination */}
        <div className='mb-6'>
          {/* Search Bar with Toggle and Pagination */}
          <div className='flex items-center justify-between gap-4 mb-4'>
            <div className='flex-1'>
              <ModSearch
                onSearch={handleSearch}
                availableTags={[]}
                isExpanded={showModSearch}
                onToggleExpanded={() => setShowModSearch(!showModSearch)}
                totalModules={state.n}
              />
            </div>
            
            {/* Pagination Controls beside search */}
            {!state.loading && state.n > 0 && (
              <PaginationControls />
            )}
          </div>
          
          {/* Quick Stats when search is expanded */}
          {showModSearch && (
            <div className='bg-black/40 border border-green-500/30 rounded p-4 mb-4'>
              <h3 className='text-green-500 text-sm font-bold mb-3 uppercase'>Module Stats</h3>
              <div className='grid grid-cols-2 md:grid-cols-4 gap-4 text-xs'>
                <div>
                  <span className='text-green-500/70'>Total Modules:</span>
                  <span className='text-green-500 ml-2'>{state.n}</span>
                </div>
                <div>
                  <span className='text-green-500/70'>Filtered:</span>
                  <span className='text-green-500 ml-2'>{state.modules.length}</span>
                </div>
                <div>
                  <span className='text-green-500/70'>Showing:</span>
                  <span className='text-green-500 ml-2'>{state.modules.length}</span>
                </div>
                <div>
                  <span className='text-green-500/70'>Page:</span>
                  <span className='text-green-500 ml-2'>{page}/{totalPages || 1}</span>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Empty State */}
        {!state.loading && state.modules.length === 0 && (
          <div className='py-8 text-center text-green-500' role='status'>
            {searchFilters.searchTerm || searchFilters.includeTerms.length > 0 || searchFilters.excludeTerms.length > 0
              ? 'NO MODULES MATCH YOUR FILTERS.'
              : 'NO MODULES AVAILABLE.'}
          </div>
        )}

        {/* Modules Grid
        {state.loading && (
          <div className='flex items-center justify-center h-64'>
            <Loading/>
          </div>
        )} */}

        {/* Modules List */}
        {!state.loading && state.modules.length > 0 && 
        <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' role='list'>
          {state.modules.map((m) => (
            <div key={m.key} role='listitem'>
              <ModuleCard module={m} />
            </div>
          ))}
        </div>
        }
      </main>

      <Footer />
    </div>
  )
}