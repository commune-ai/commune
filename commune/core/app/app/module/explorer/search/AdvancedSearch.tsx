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
    <div className="w-full space-y-2 font-mono">
      {/* Main Search Bar */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            placeholder="SEARCH MODULES..."
            value={filters.searchTerm}
            onChange={(e) => handleSearchTermChange(e.target.value)}
            className="w-full px-3 py-2 bg-black border border-green-500 text-green-500 placeholder-green-500/50 focus:outline-none uppercase"
          />
        </div>
        
        {/* Advanced Filter Toggle */}
        <button
          onClick={onToggleExpanded}
          className={`px-3 py-2 bg-black border border-green-500 text-green-500 hover:bg-green-500 hover:text-black transition-colors ${
            hasActiveFilters ? 'bg-green-500/20' : ''
          }`}
          aria-label="Toggle advanced filters"
          title="Advanced search"
        >
          <Filter size={18} className={isExpanded ? 'rotate-180' : ''} />
        </button>
      </div>

      {/* Advanced Filters Panel */}
      {isExpanded && (
        <div className="border border-green-500 bg-black p-4 space-y-4">
          {/* Active Filters Summary */}
          {hasActiveFilters && (
            <div className="flex items-center justify-between border-b border-green-500/30 pb-2">
              <span className="text-green-500 text-sm">ACTIVE FILTERS</span>
              <button
                onClick={clearAllFilters}
                className="text-green-500 hover:text-green-400 text-sm"
              >
                [CLEAR ALL]
              </button>
            </div>
          )}

          {/* Include Tags */}
          <div className="space-y-2">
            <label className="text-green-500 text-sm flex items-center gap-2">
              <Tag size={14} />
              INCLUDE TAGS
            </label>
            <div className="flex flex-wrap gap-2">
              {filters.includeTags.map(tag => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-2 py-1 bg-green-500/20 border border-green-500 text-green-400 text-xs"
                >
                  #{tag}
                  <button
                    onClick={() => removeFromFilter('includeTags', tag)}
                    className="hover:text-green-300"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
              <div className="flex items-center gap-1">
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
                  className="px-2 py-1 bg-black border border-green-500/50 text-green-500 placeholder-green-500/30 focus:outline-none text-xs uppercase w-24"
                />
                <button
                  onClick={() => addToFilter('includeTags', tagInput)}
                  className="p-1 border border-green-500/50 text-green-500 hover:bg-green-500 hover:text-black"
                >
                  <Plus size={12} />
                </button>
              </div>
            </div>
          </div>

          {/* Exclude Tags */}
          <div className="space-y-2">
            <label className="text-green-500 text-sm flex items-center gap-2">
              <Tag size={14} className="opacity-50" />
              EXCLUDE TAGS
            </label>
            <div className="flex flex-wrap gap-2">
              {filters.excludeTags.map(tag => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-2 py-1 bg-red-500/20 border border-red-500 text-red-400 text-xs"
                >
                  #{tag}
                  <button
                    onClick={() => removeFromFilter('excludeTags', tag)}
                    className="hover:text-red-300"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
              <div className="flex items-center gap-1">
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
                  className="px-2 py-1 bg-black border border-red-500/50 text-red-500 placeholder-red-500/30 focus:outline-none text-xs uppercase w-24"
                />
                <button
                  onClick={() => {
                    addToFilter('excludeTags', tagInput)
                    setActiveFilter(null)
                  }}
                  className="p-1 border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-black"
                >
                  <Plus size={12} />
                </button>
              </div>
            </div>
          </div>

          {/* Include Terms */}
          <div className="space-y-2">
            <label className="text-green-500 text-sm flex items-center gap-2">
              <Hash size={14} />
              INCLUDE TERMS
            </label>
            <div className="flex flex-wrap gap-2">
              {filters.includeTerms.map(term => (
                <span
                  key={term}
                  className="inline-flex items-center gap-1 px-2 py-1 bg-green-500/20 border border-green-500 text-green-400 text-xs"
                >
                  "{term}"
                  <button
                    onClick={() => removeFromFilter('includeTerms', term)}
                    className="hover:text-green-300"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
              <div className="flex items-center gap-1">
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
                  className="px-2 py-1 bg-black border border-green-500/50 text-green-500 placeholder-green-500/30 focus:outline-none text-xs uppercase w-32"
                />
                <button
                  onClick={() => {
                    addToFilter('includeTerms', termInput)
                    setActiveFilter(null)
                  }}
                  className="p-1 border border-green-500/50 text-green-500 hover:bg-green-500 hover:text-black"
                >
                  <Plus size={12} />
                </button>
              </div>
            </div>
          </div>

          {/* Exclude Terms */}
          <div className="space-y-2">
            <label className="text-green-500 text-sm flex items-center gap-2">
              <Hash size={14} className="opacity-50" />
              EXCLUDE TERMS
            </label>
            <div className="flex flex-wrap gap-2">
              {filters.excludeTerms.map(term => (
                <span
                  key={term}
                  className="inline-flex items-center gap-1 px-2 py-1 bg-red-500/20 border border-red-500 text-red-400 text-xs"
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
              <div className="flex items-center gap-1">
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
                  className="px-2 py-1 bg-black border border-red-500/50 text-red-500 placeholder-red-500/30 focus:outline-none text-xs uppercase w-32"
                />
                <button
                  onClick={() => {
                    addToFilter('excludeTerms', termInput)
                    setActiveFilter(null)
                  }}
                  className="p-1 border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-black"
                >
                  <Plus size={12} />
                </button>
              </div>
            </div>
          </div>

          {/* Quick Tag Suggestions */}
          {availableTags.length > 0 && (
            <div className="pt-2 border-t border-green-500/30">
              <label className="text-green-500 text-sm">POPULAR TAGS</label>
              <div className="flex flex-wrap gap-2 mt-2">
                {availableTags.slice(0, 10).map(tag => (
                  <button
                    key={tag}
                    onClick={() => addToFilter('includeTags', tag)}
                    className="px-2 py-1 bg-black border border-green-500/30 text-green-500/70 hover:border-green-500 hover:text-green-500 text-xs"
                  >
                    #{tag}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Pagination Controls at Bottom */}
          {onPageChange && totalPages > 1 && (
            <div className="pt-4 mt-4 border-t border-green-500/30">
              <div className="flex items-center justify-between">
                <span className="text-green-500 text-sm">PAGE NAVIGATION</span>
                <nav className="flex items-center gap-2 font-mono text-sm">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage <= 1}
                    className="px-2 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-none"
                    aria-label="Previous page"
                  >
                    <ChevronLeft size={14} />
                  </button>
                  
                  <span className="px-2 text-green-500 text-xs">
                    {currentPage}/{totalPages}
                  </span>
                  
                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage >= totalPages}
                    className="px-2 py-1 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed uppercase transition-none"
                    aria-label="Next page"
                  >
                    <ChevronRight size={14} />
                  </button>
                </nav>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default AdvancedSearch