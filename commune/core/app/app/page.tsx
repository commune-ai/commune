'use client'

import { useEffect, useState, useMemo, useCallback } from 'react'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import ModuleCard from '@/app/search/ModuleCard'
import { CreateModule } from '@/app/search/ModuleCreate'
import { ModuleType } from '@/app/types/module'
import {Footer} from '@/app/components/Footer'
import { useSearchContext } from '@/app/context/SearchContext'
interface ModulesState {
  modules: ModuleType[]
  n: number
  loading: boolean
  error: string | null
}

export default function Modules() {
  const client = new Client()
  const { searchFilters, updateSearchFilters } = useSearchContext()
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [state, setState] = useState<ModulesState>({
    modules: [],
    n: 0,
    loading: false,
    error: null
  })

  // Use search context values
  const page = searchFilters.page || 1
  const pageSize = searchFilters.pageSize || 20
  const totalPages = Math.ceil(state.n / pageSize)

  const fetchModules = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    let params: any = {}
    if (searchFilters.searchTerm) {
      params.search = searchFilters.searchTerm
    }

    try {
      // Fetch modules for current page
      const n: number = await client.call('n', params)
      let modulesData: ModuleType[] = []
      if (n > 0) {
        // add the page and pageSize to params
        params.page = page
        params.page_size = pageSize
        modulesData = await client.call('modules', params)
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
    if (newPage < 1 || newPage > totalPages) return
    updateSearchFilters({ page: newPage })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleCreateSuccess = () => {
    setShowCreateForm(false)
    setState(prev => ({ ...prev, error: null }))
    fetchModules() // Refresh modules
  }

  useEffect(() => {
    fetchModules()
  }, [searchFilters.searchTerm, page])

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
      <main className='p-2 pt-1' role='main'>
        {/* Header Info */}
        <div className='mb-6 flex items-center justify-between'>
            <p className='text-green-500/70'>
              {state.n > 0 ? `${state.n} MODULES FOUND` : 'SEARCHING...'}
              {searchFilters.searchTerm && ` FOR "${searchFilters.searchTerm.toUpperCase()}"`}
            </p>
          
          {/* Pagination */}
          {totalPages > 1 && (
            <div className='flex items-center gap-2'>
              <button
                onClick={() => handlePageChange(page - 1)}
                disabled={page <= 1}
                className='px-3 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed'
              >
                PREV
              </button>
              <span className='px-3 text-green-500'>
                {page} / {totalPages}
              </span>
              <button
                onClick={() => handlePageChange(page + 1)}
                disabled={page >= totalPages}
                className='px-3 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed'
              >
                NEXT
              </button>
            </div>
          )}
        </div>
        
        {/* Loading State */}
        {state.loading && (
          <div className='py-8 text-center'>
            <Loading />
          </div>
        )}

        {/* Empty State */}
        {!state.loading && state.modules.length === 0 && (
          <div className='py-8 text-center text-green-500' role='status'>
            {searchFilters.searchTerm
              ? 'NO MODULES MATCH YOUR SEARCH.'
              : 'NO MODULES AVAILABLE.'}
          </div>
        )}

        {/* Modules Grid */}
        {!state.loading && state.modules.length > 0 && (
          <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' role='list'>
            {state.modules.map((m) => (
              <div key={m.key} role='listitem'>
                <ModuleCard module={m} />
              </div>
            ))}
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}