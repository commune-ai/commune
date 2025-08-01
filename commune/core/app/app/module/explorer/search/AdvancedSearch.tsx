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
}

interface AdvancedSearchProps {
  onSearch: (filters: SearchFilters) => void
  availableTags?: string[]
  isExpanded: boolean
  onToggleExpanded: () => void
  // Pagination props
  currentPage?: number
  totalPages?: number
  onPageChange?: (page: number) => void
}

export const AdvancedSearch = ({ 
  onSearch, 
  availableTags = [], 
  isExpanded,
  onToggleExpanded,
  currentPage = 1,
  totalPages = 1,
  onPageChange
}: AdvancedSearchProps) => {
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
    <div className="w-full max-w-4xl mx-auto space-y-3 font-mono">
      {/* Main Search Bar */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            placeholder="SEARCH MODULES..."
            value={filters.searchTerm}
            onChange={(e) => handleSearchTermChange(e.target.value)}
            className="w-full px-4 py-3 bg-black/80 border-2 border-white rounded-full text-white placeholder-white/50 focus:outline-none focus:border-white/80 focus:shadow-[0_0_20px_rgba(255,255,255,0.3)] uppercase transition-all"
          />
          <Search className="absolute right-4 top-1/2 -translate-y-1/2 text-white/50" size={20} />
        </div>
        
        {/* Advanced Filter Toggle */}
        <button
          onClick={onToggleExpanded}
          className={`px-4 py-3 bg-black/80 border-2 border-white rounded-full text-white hover:bg-white hover:text-black transition-all ${
            hasActiveFilters ? 'bg-white/20' : ''
          }`}
          aria-label="Toggle advanced filters"
          title="Advanced search"
        >
          <Filter size={20} className={isExpanded ? 'rotate-180' : ''} />
        </button>
      </div>

      {/* Advanced Filters Panel - Horizontal Layout */}
      {isExpanded && (
        <div className="border-2 border-white bg-black/90 rounded-2xl p-6 shadow-[0_0_30px_rgba(255,255,255,0.2)] overflow-x-auto">
          {/* Active Filters Summary */}
          {hasActiveFilters && (
            <div className="flex items-center justify-between border-b border-white/30 pb-3 mb-4">
              <span className="text-white text-base font-semibold uppercase">Active Filters</span>
              <button
                onClick={clearAllFilters}
                className="text-white hover:text-white/80 text-sm px-3 py-1 border border-white/50 rounded-full hover:border-white/80 transition-all"
              >
                CLEAR ALL
              </button>
            </div>
          )}

          {/* Horizontal Filter Sections */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 min-w-0">
            {/* Include Tags */}
            <div className="min-w-0">
              <label className="text-white text-sm flex items-center gap-2 font-semibold mb-3">
                <Tag size={16} />
                INCLUDE TAGS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[32px]">
                  {filters.includeTags.map(tag => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-white/20 border border-white rounded-full text-white text-sm hover:bg-white/30 transition-all"
                    >
                      #{tag}
                      <button
                        onClick={() => removeFromFilter('includeTags', tag)}
                        className="hover:text-white/70"
                      >
                        <X size={14} />
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
                    className="flex-1 px-3 py-1.5 bg-black border border-white/50 rounded-full text-white placeholder-white/30 focus:outline-none focus:border-white text-sm uppercase"
                  />
                  <button
                    onClick={() => addToFilter('includeTags', tagInput)}
                    className="p-1.5 border border-white/50 rounded-full text-white hover:bg-white hover:text-black transition-all"
                  >
                    <Plus size={14} />
                  </button>
                </div>
              </div>
            </div>

            {/* Exclude Tags */}
            <div className="min-w-0">
              <label className="text-white text-sm flex items-center gap-2 font-semibold mb-3">
                <Tag size={16} className="opacity-50" />
                EXCLUDE TAGS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[32px]">
                  {filters.excludeTags.map(tag => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-red-500/20 border border-red-500 rounded-full text-red-400 text-sm hover:bg-red-500/30 transition-all"
                    >
                      #{tag}
                      <button
                        onClick={() => removeFromFilter('excludeTags', tag)}
                        className="hover:text-red-300"
                      >
                        <X size={14} />
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
                    className="flex-1 px-3 py-1.5 bg-black border border-red-500/50 rounded-full text-red-500 placeholder-red-500/30 focus:outline-none focus:border-red-400 text-sm uppercase"
                  />
                  <button
                    onClick={() => {
                      addToFilter('excludeTags', tagInput)
                      setActiveFilter(null)
                    }}
                    className="p-1.5 border border-red-500/50 rounded-full text-red-500 hover:bg-red-500 hover:text-black transition-all"
                  >
                    <Plus size={14} />
                  </button>
                </div>
              </div>
            </div>

            {/* Include Terms */}
            <div className="min-w-0">
              <label className="text-white text-sm flex items-center gap-2 font-semibold mb-3">
                <Hash size={16} />
                INCLUDE TERMS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[32px]">
                  {filters.includeTerms.map(term => (
                    <span
                      key={term}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-white/20 border border-white rounded-full text-white text-sm hover:bg-white/30 transition-all"
                    >
                      "{term}"
                      <button
                        onClick={() => removeFromFilter('includeTerms', term)}
                        className="hover:text-white/70"
                      >
                        <X size={14} />
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
                    className="flex-1 px-3 py-1.5 bg-black border border-white/50 rounded-full text-white placeholder-white/30 focus:outline-none focus:border-white text-sm uppercase"
                  />
                  <button
                    onClick={() => {
                      addToFilter('includeTerms', termInput)
                      setActiveFilter(null)
                    }}
                    className="p-1.5 border border-white/50 rounded-full text-white hover:bg-white hover:text-black transition-all"
                  >
                    <Plus size={14} />
                  </button>
                </div>
              </div>
            </div>

            {/* Exclude Terms */}
            <div className="min-w-0">
              <label className="text-white text-sm flex items-center gap-2 font-semibold mb-3">
                <Hash size={16} className="opacity-50" />
                EXCLUDE TERMS
              </label>
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 min-h-[32px]">
                  {filters.excludeTerms.map(term => (
                    <span
                      key={term}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-red-500/20 border border-red-500 rounded-full text-red-400 text-sm hover:bg-red-500/30 transition-all"
                    >
                      "{term}"
                      <button
                        onClick={() => removeFromFilter('excludeTerms', term)}
                        className="hover:text-red-300"
                      >
                        <X size={14} />
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
                        addToFilter('excludeTerms', termInput)
                        setActiveFilter(null)
                      }
                    }}
                    placeholder="EXCLUDE TERM"
                    className="flex-1 px-3 py-1.5 bg-black border border-red-500/50 rounded-full text-red-500 placeholder-red-500/30 focus:outline-none focus:border-red-400 text-sm uppercase"
                  />
                  <button
                    onClick={() => {
                      addToFilter('excludeTerms', termInput)
                      setActiveFilter(null)
                    }}
                    className="p-1.5 border border-red-500/50 rounded-full text-red-500 hover:bg-red-500 hover:text-black transition-all"
                  >
                    <Plus size={14} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Section - Popular Tags and Pagination in a row */}
          <div className="mt-6 pt-4 border-t border-white/30">
            <div className="flex flex-wrap items-center justify-between gap-4">
              {/* Quick Tag Suggestions */}
              {availableTags.length > 0 && (
                <div className="flex-1">
                  <label className="text-white text-sm font-semibold mb-2 block">POPULAR TAGS</label>
                  <div className="flex flex-wrap gap-2">
                    {availableTags.slice(0, 8).map(tag => (
                      <button
                        key={tag}
                        onClick={() => addToFilter('includeTags', tag)}
                        className="px-3 py-1 bg-black border border-white/30 rounded-full text-white/70 hover:border-white hover:text-white text-sm transition-all"
                      >
                        #{tag}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Pagination Controls */}
              {onPageChange && totalPages > 1 && (
                <nav className="flex items-center gap-3 font-mono text-sm">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage <= 1}
                    className="p-1.5 border border-white rounded-full text-white hover:bg-white hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-all"
                    aria-label="Previous page"
                  >
                    <ChevronLeft size={16} />
                  </button>
                  
                  <span className="px-3 text-white">
                    {currentPage}/{totalPages}
                  </span>
                  
                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage >= totalPages}
                    className="p-1.5 border border-white rounded-full text-white hover:bg-white hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-all"
                    aria-label="Next page"
                  >
                    <ChevronRight size={16} />
                  </button>
                </nav>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AdvancedSearch