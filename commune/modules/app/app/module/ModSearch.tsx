'use client'

import { useState, useCallback, useEffect } from 'react'
import { X, Plus, Search, Filter, Tag, Hash, ChevronLeft, ChevronRight } from 'lucide-react'
import { debounce } from 'lodash'

export interface SearchFilters {
  searchTerm: string
  includeTags: string[]
  excludeTags: string[]
  includeTerms: string[]
  excludeTerms: string[]
  page?: number
  pageSize?: number
}

interface ModSearchProps {
  onSearch: (filters: SearchFilters) => void
  availableTags?: string[]
  isExpanded: boolean
  onToggleExpanded: () => void
  // Pagination props
  page?: number
  totalPages?: number
  onPageChange?: (page: number) => void
}

export const ModSearch = ({ 
  onSearch, 
  availableTags = [], 
  isExpanded,
  onToggleExpanded,
  page = 1,
  totalPages = 1,
  onPageChange
}: ModSearchProps) => {
  const [filters, setFilters] = useState<SearchFilters>({
    searchTerm: '',
    includeTags: [],
    excludeTags: [],
    includeTerms: [],
    excludeTerms: []
  })

  const [tagInput, setTagInput] = useState('')
  const [termInput, setTermInput] = useState('')
  const [activeFilter, setActiveFilter] = useState<'includeTags' | 'excludeTags' | 'includeTerms' | 'excludeTerms' | null>(null)

  // Debounced search
  const debouncedSearch = useCallback(
    debounce((searchFilters: SearchFilters) => {
      onSearch(searchFilters)
    }, 300),
    [onSearch]
  )

  // Update search when filters change
  useEffect(() => {
    debouncedSearch(filters)
  }, [filters, debouncedSearch])

  const handleSearchTermChange = (value: string) => {
    setFilters(prev => ({ ...prev, searchTerm: value }))
  }

  const addToFilter = (filterType: keyof SearchFilters, value: string) => {
    if (!value.trim()) return
    
    const normalizedValue = value.toLowerCase().trim()
    if (Array.isArray(filters[filterType]) && !filters[filterType].includes(normalizedValue)) {
      setFilters(prev => ({
        ...prev,
        [filterType]: [...prev[filterType], normalizedValue]
      }))
    }
    
    // Clear inputs
    if (filterType.includes('Tags')) {
      setTagInput('')
    } else {
      setTermInput('')
    }
  }

  const removeFromFilter = (filterType: keyof SearchFilters, value: string) => {
    if (Array.isArray(filters[filterType])) {
      setFilters(prev => ({
        ...prev,
        [filterType]: prev[filterType].filter(item => item !== value)
      }))
    }
  }

  const clearAllFilters = () => {
    setFilters({
      searchTerm: '',
      includeTags: [],
      excludeTags: [],
      includeTerms: [],
      excludeTerms: []
    })
    setTagInput('')
    setTermInput('')
  }

  const hasActiveFilters = filters.includeTags.length > 0 || 
                          filters.excludeTags.length > 0 || 
                          filters.includeTerms.length > 0 || 
                          filters.excludeTerms.length > 0

  const handlePageChange = (newPage: number) => {
    if (onPageChange && newPage >= 1 && newPage <= totalPages) {
      onPageChange(newPage)
    }
  }

  return (
    <div className="w-full max-w-6xl mx-auto space-y-4 font-mono">
      {/* Main Search Bar with Pagination */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <input
            type="text"
            placeholder="SEARCH MODULES..."
            value={filters.searchTerm}
            onChange={(e) => handleSearchTermChange(e.target.value)}
            className="w-full px-4 py-2.5 bg-black border-2 border-green-500 text-green-500 placeholder-green-500/50 focus:outline-none focus:border-green-400 focus:shadow-[0_0_15px_rgba(34,197,94,0.3)] uppercase transition-all"
          />
          <Search className="absolute right-4 top-1/2 -translate-y-1/2 text-green-500/50" size={18} />
        </div>
        
        {/* Advanced Filter Toggle */}
        <button
          onClick={onToggleExpanded}
          className={`px-4 py-2.5 bg-black border-2 border-green-500 text-green-500 hover:bg-green-500 hover:text-black transition-all ${
            hasActiveFilters ? 'bg-green-500/20' : ''
          }`}
          aria-label="Toggle advanced filters"
          title="Advanced search"
        >
          <Filter size={18} className={isExpanded ? 'rotate-180' : ''} />
        </button>

        {/* Sleek Pagination Controls */}
        {onPageChange && totalPages > 1 && (
          <nav className="flex items-center gap-1 font-mono text-sm border-2 border-green-500 bg-black px-1 py-1">
            <button
              onClick={() => handlePageChange(page - 1)}
              disabled={page <= 1}
              className="p-1.5 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              aria-label="Previous page"
            >
              <ChevronLeft size={16} />
            </button>
            
            <span className="px-3 text-green-500 min-w-[80px] text-center">
              {page}/{totalPages}
            </span>
            
            <button
              onClick={() => handlePageChange(page + 1)}
              disabled={page >= totalPages}
              className="p-1.5 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              aria-label="Next page"
            >
              <ChevronRight size={16} />
            </button>
          </nav>
        )}
      </div>

      {/* Advanced Filters Panel - Horizontal Layout */}
      {isExpanded && (
        <div className="border-2 border-green-500 bg-black p-6 shadow-[0_0_20px_rgba(34,197,94,0.2)]">
          {/* Active Filters Summary */}
          {hasActiveFilters && (
            <div className="flex items-center justify-between border-b border-green-500/30 pb-3 mb-4">
              <span className="text-green-500 text-sm font-bold uppercase">Active Filters</span>
              <button
                onClick={clearAllFilters}
                className="text-green-500 hover:bg-green-500 hover:text-black text-xs px-3 py-1 border border-green-500 transition-all"
              >
                CLEAR ALL
              </button>
            </div>
          )}

          {/* Horizontal Filter Sections */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
            {/* Include Tags */}
            <div>
              <label className="text-green-500 text-xs flex items-center gap-2 font-bold mb-3 uppercase">
                <Tag size={14} />
                INCLUDE TAGS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[28px]">
                  {filters.includeTags.map(tag => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-green-500/20 border border-green-500 text-green-500 text-xs hover:bg-green-500/30 transition-all"
                    >
                      #{tag}
                      <button
                        onClick={() => removeFromFilter('includeTags', tag)}
                        className="hover:text-green-400"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToFilter('includeTags', tagInput)
                      }
                    }}
                    placeholder="ADD TAG"
                    className="flex-1 px-3 py-1 bg-black border border-green-500/50 text-green-500 placeholder-green-500/30 focus:outline-none focus:border-green-500 text-xs uppercase"
                  />
                  <button
                    onClick={() => addToFilter('includeTags', tagInput)}
                    className="p-1 border border-green-500/50 text-green-500 hover:bg-green-500 hover:text-black transition-all"
                  >
                    <Plus size={12} />
                  </button>
                </div>
              </div>
            </div>

            {/* Exclude Tags */}
            <div>
              <label className="text-red-500 text-xs flex items-center gap-2 font-bold mb-3 uppercase">
                <Tag size={14} className="opacity-70" />
                EXCLUDE TAGS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[28px]">
                  {filters.excludeTags.map(tag => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-red-500/20 border border-red-500 text-red-500 text-xs hover:bg-red-500/30 transition-all"
                    >
                      #{tag}
                      <button
                        onClick={() => removeFromFilter('excludeTags', tag)}
                        className="hover:text-red-400"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={activeFilter === 'excludeTags' ? tagInput : ''}
                    onChange={(e) => {
                      setActiveFilter('excludeTags')
                      setTagInput(e.target.value)
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToFilter('excludeTags', tagInput)
                        setActiveFilter(null)
                      }
                    }}
                    placeholder="EXCLUDE TAG"
                    className="flex-1 px-3 py-1 bg-black border border-red-500/50 text-red-500 placeholder-red-500/30 focus:outline-none focus:border-red-500 text-xs uppercase"
                  />
                  <button
                    onClick={() => {
                      addToFilter('excludeTags', tagInput)
                      setActiveFilter(null)
                    }}
                    className="p-1 border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-black transition-all"
                  >
                    <Plus size={12} />
                  </button>
                </div>
              </div>
            </div>

            {/* Include Terms */}
            <div>
              <label className="text-green-500 text-xs flex items-center gap-2 font-bold mb-3 uppercase">
                <Hash size={14} />
                INCLUDE TERMS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[28px]">
                  {filters.includeTerms.map(term => (
                    <span
                      key={term}
                      className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-green-500/20 border border-green-500 text-green-500 text-xs hover:bg-green-500/30 transition-all"
                    >
                      "{term}"
                      <button
                        onClick={() => removeFromFilter('includeTerms', term)}
                        className="hover:text-green-400"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={activeFilter === 'includeTerms' ? termInput : ''}
                    onChange={(e) => {
                      setActiveFilter('includeTerms')
                      setTermInput(e.target.value)
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToFilter('includeTerms', termInput)
                        setActiveFilter(null)
                      }
                    }}
                    placeholder="INCLUDE TERM"
                    className="flex-1 px-3 py-1 bg-black border border-green-500/50 text-green-500 placeholder-green-500/30 focus:outline-none focus:border-green-500 text-xs uppercase"
                  />
                  <button
                    onClick={() => {
                      addToFilter('includeTerms', termInput)
                      setActiveFilter(null)
                    }}
                    className="p-1 border border-green-500/50 text-green-500 hover:bg-green-500 hover:text-black transition-all"
                  >
                    <Plus size={12} />
                  </button>
                </div>
              </div>
            </div>

            {/* Exclude Terms */}
            <div>
              <label className="text-red-500 text-xs flex items-center gap-2 font-bold mb-3 uppercase">
                <Hash size={14} className="opacity-70" />
                EXCLUDE TERMS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[28px]">
                  {filters.excludeTerms.map(term => (
                    <span
                      key={term}
                      className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-red-500/20 border border-red-500 text-red-500 text-xs hover:bg-red-500/30 transition-all"
                    >
                      "{term}"
                      <button
                        onClick={() => removeFromFilter('excludeTerms', term)}
                        className="hover:text-red-300"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={activeFilter === 'excludeTerms' ? termInput : ''}
                    onChange={(e) => {
                      setActiveFilter('excludeTerms')
                      setTermInput(e.target.value)
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addTilter('excludeTerms', termInput)
                        setActiveFilter(null)
                      }
                    }}
                    placeholder="EXCLUDE TERM"
                    className="flex-1 px-3 py-1 bg-black border border-red-500/50 text-red-500 placeholder-red-500/30 focus:outline-none focus:border-red-500 text-xs uppercase"
                  />
                  <button
                    onClick={() => {
                      addToFilter('excludeTerms', termInput)
                      setActiveFilter(null)
                    }}
                    className="p-1 border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-black transition-all"
                  >
                    <Plus size={12} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Section - Popular Tags */}
          {availableTags.length > 0 && (
            <div className="mt-6 pt-4 border-t border-green-500/30">
              <label className="text-green-500 text-xs font-bold mb-2 block uppercase">POPULAR TAGS</label>
              <div className="flex flex-wrap gap-2">
                {availableTags.slice(0, 10).map(tag => (
                  <button
                    key={tag}
                    onClick={() => addToFilter('includeTags', tag)}
                    className="px-3 py-1 bg-black border border-green-500/30 text-green-500/70 hover:border-green-500 hover:text-green-500 text-xs transition-all uppercase"
                  >
                    #{tag}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ModSearch