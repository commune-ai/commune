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
  const  pageSize = 10
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

  const fetchModules = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      // Fetch modules for current page
      const n : number = await client.call('n', { search: searchFilters.searchTerm })
      let modulesData: ModuleType[] = []
      if (n > 0 ) {
         modulesData = await client.call('modules', { page_size: pageSize, page: page , search: searchFilters.searchTerm })
      }
      setState({
        modules: modulesData,
        n,
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
    if (newPage < 1 || newPage > state.totalPages) return
    setPage(newPage)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleSearch = useCallback((filters: SearchFilters) => {
    setSearchFilters(filters)
    setPage(1) // Reset to first page on new search
  }, [])

  const handleCreateSuccess = () => {
    setShowCreateForm(false)
    setState(prev => ({ ...prev, error: null }))
  }

  useEffect(() => {
    fetchModules()
  }, [page, searchFilters])

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
                page={page}
                totalPages={totalPages}
                onPageChange={handlePageChange}
              />
            </div>
          </div>
          
        </div>
        
        {/* Empty State */}
        {!state.loading && state.modules.length === 0 && (
          <div className='py-8 text-center text-green-500' role='status'>
            {searchFilters.searchTerm || searchFilters.includeTerms.length > 0 || searchFilters.excludeTerms.length > 0
              ? 'NO MODULES MATCH YOUR FILTERS.'
              : 'NO MODULES AVAILABLE.'}
          </div>
        )}

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