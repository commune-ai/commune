'use client'

import { useState, useCallback, useEffect } from 'react'
import { ModSearch, SearchFilters } from './ModSearch'
import { CreateModule } from '../module/ModuleCreate'
import { Client } from '@/app/client/client'
import { ModuleType } from '@/app/types/module'
import { Plus, Search, Terminal, Code, Globe, Tag } from 'lucide-react'

interface SearchResult {
  modules: ModuleType[]
  total: number
  page: number
  pageSize: number
}

export default function ModuleSearchPage() {
  const [searchResults, setSearchResults] = useState<SearchResult>({
    modules: [],
    total: 0,
    page: 1,
    pageSize: 20
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [isExpanded, setIsExpanded] = useState(false)
  const [showCreate, setShowCreate] = useState(false)
  const [currentFilters, setCurrentFilters] = useState<SearchFilters>({
    searchTerm: '',
    includeTags: [],
    excludeTags: [],
    includeTerms: [],
    excludeTerms: [],
    page: 1,
    pageSize: 20
  })
  
  const client = new Client()

  // Fetch modules based on filters
  const fetchModules = useCallback(async (filters: SearchFilters) => {
    setLoading(true)
    setError('')
    
    try {
      const response = await client.call('search_modules', {
        ...filters,
        page: filters.page || 1,
        pageSize: filters.pageSize || 20
      })
      
      setSearchResults({
        modules: response.modules || [],
        total: response.total || 0,
        page: response.page || 1,
        pageSize: response.pageSize || 20
      })
    } catch (err: any) {
      setError(err.message || 'Failed to fetch modules')
      setSearchResults({ modules: [], total: 0, page: 1, pageSize: 20 })
    } finally {
      setLoading(false)
    }
  }, [])

  // Handle search from ModSearch component
  const handleSearch = useCallback((filters: SearchFilters) => {
    setCurrentFilters(filters)
    fetchModules(filters)
  }, [fetchModules])

  // Handle page change
  const handlePageChange = useCallback((page: number) => {
    const newFilters = { ...currentFilters, page }
    setCurrentFilters(newFilters)
    fetchModules(newFilters)
  }, [currentFilters, fetchModules])

  // Handle successful module creation
  const handleCreateSuccess = useCallback(() => {
    setShowCreate(false)
    // Refresh the search results
    fetchModules(currentFilters)
  }, [currentFilters, fetchModules])

  // Initial load
  useEffect(() => {
    fetchModules(currentFilters)
  }, [])

  const totalPages = Math.ceil(searchResults.total / searchResults.pageSize)

  return (
    <div className="min-h-screen bg-black text-green-500 font-mono p-8">
      {/* Terminal Header */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Terminal className="text-green-500" size={32} />
            <h1 className="text-3xl font-bold uppercase tracking-wider">MODULE REGISTRY</h1>
          </div>
          
          {/* Create Module Button */}
          <button
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-2 px-4 py-2 bg-black border-2 border-green-500 text-green-500 hover:bg-green-500 hover:text-black transition-all uppercase font-bold"
          >
            <Plus size={18} />
            CREATE MODULE
          </button>
        </div>
        
        <div className="text-green-500/70 text-sm">
          <p>// COMMUNE MODULE SEARCH INTERFACE v2.0</p>
          <p>// {searchResults.total} MODULES INDEXED</p>
        </div>
      </div>

      {/* Search Component */}
      <ModSearch
        onSearch={handleSearch}
        isExpanded={isExpanded}
        onToggleExpanded={() => setIsExpanded(!isExpanded)}
        page={currentFilters.page || 1}
        totalPages={totalPages}
        onPageChange={handlePageChange}
      />

      {/* Loading State */}
      {loading && (
        <div className="max-w-6xl mx-auto mt-8 text-center">
          <div className="inline-flex items-center gap-3 text-green-500">
            <div className="animate-spin h-6 w-6 border-2 border-green-500 border-t-transparent rounded-full" />
            <span className="uppercase">SEARCHING MODULES...</span>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="max-w-6xl mx-auto mt-8 p-4 border-2 border-red-500 bg-red-500/10">
          <p className="text-red-500 uppercase">ERROR: {error}</p>
        </div>
      )}

      {/* Results Grid */}
      {!loading && searchResults.modules.length > 0 && (
        <div className="max-w-6xl mx-auto mt-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {searchResults.modules.map((module) => (
              <ModuleCard key={module.name} module={module} />
            ))}
          </div>
        </div>
      )}

      {/* No Results */}
      {!loading && searchResults.modules.length === 0 && !error && (
        <div className="max-w-6xl mx-auto mt-16 text-center">
          <Search className="mx-auto text-green-500/30 mb-4" size={64} />
          <p className="text-green-500/70 uppercase">NO MODULES FOUND</p>
          <p className="text-green-500/50 text-sm mt-2">TRY ADJUSTING YOUR SEARCH FILTERS</p>
        </div>
      )}

      {/* Create Module Modal */}
      {showCreate && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-8 z-50">
          <CreateModule
            onClose={() => setShowCreate(false)}
            onSuccess={handleCreateSuccess}
          />
        </div>
      )}
    </div>
  )
}

// Module Card Component
function ModuleCard({ module }: { module: ModuleType }) {
  return (
    <div className="border-2 border-green-500/30 bg-black hover:border-green-500 transition-all p-4 space-y-3">
      {/* Module Name */}
      <div className="flex items-center justify-between">
        <h3 className="text-green-400 font-bold uppercase truncate">{module.name}</h3>
        <Code className="text-green-500/50" size={16} />
      </div>

      {/* Module Key */}
      {module.key && (
        <p className="text-green-500/70 text-xs font-mono truncate">KEY: {module.key}</p>
      )}

      {/* Description */}
      {module.desc && (
        <p className="text-green-500/60 text-sm line-clamp-2">{module.desc}</p>
      )}

      {/* Tags */}
      {module.tags && module.tags.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <Tag className="text-green-500/50" size={12} />
          {module.tags.slice(0, 3).map((tag, idx) => (
            <span key={idx} className="text-xs px-2 py-0.5 border border-green-500/30 text-green-500/70">
              #{tag}
            </span>
          ))}
          {module.tags.length > 3 && (
            <span className="text-xs text-green-500/50">+{module.tags.length - 3}</span>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 pt-2 border-t border-green-500/20">
        {module.url && (
          <a
            href={module.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-green-500/70 hover:text-green-500 transition-colors"
          >
            <Globe size={12} />
            VISIT
          </a>
        )}
        {module.code && (
          <a
            href={module.code}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-green-500/70 hover:text-green-500 transition-colors"
          >
            <Code size={12} />
            SOURCE
          </a>
        )}
      </div>
    </div>
  )
}