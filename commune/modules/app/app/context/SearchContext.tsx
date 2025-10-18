'use client'
import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { SearchFilters } from '@/app/search/ModSearch'

interface SearchContextType {
  searchFilters: SearchFilters
  setSearchFilters: (filters: SearchFilters) => void
  updateSearchFilters: (updates: Partial<SearchFilters>) => void
  clearSearchFilters: () => void
  isSearchActive: boolean
}

const SearchContext = createContext<SearchContextType | undefined>(undefined)

export const useSearchContext = () => {
  const context = useContext(SearchContext)
  if (!context) {
    throw new Error('useSearchContext must be used within a SearchProvider')
  }
  return context
}

interface SearchProviderProps {
  children: ReactNode
}

export const SearchProvider = ({ children }: SearchProviderProps) => {
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({
    searchTerm: '',
    includeTags: [],
    excludeTags: [],
    includeTerms: [],
    excludeTerms: [],
    page: 1,
    pageSize: 20
  })

  const updateSearchFilters = useCallback((updates: Partial<SearchFilters>) => {
    setSearchFilters(prev => ({ ...prev, ...updates }))
  }, [])

  const clearSearchFilters = useCallback(() => {
    setSearchFilters({
      searchTerm: '',
      includeTags: [],
      excludeTags: [],
      includeTerms: [],
      excludeTerms: [],
      page: 1,
      pageSize: 20
    })
  }, [])

  const isSearchActive = searchFilters.searchTerm.length > 0 ||
    searchFilters.includeTags.length > 0 ||
    searchFilters.excludeTags.length > 0 ||
    searchFilters.includeTerms.length > 0 ||
    searchFilters.excludeTerms.length > 0

  return (
    <SearchContext.Provider
      value={{
        searchFilters,
        setSearchFilters,
        updateSearchFilters,
        clearSearchFilters,
        isSearchActive
      }}
    >
      {children}
    </SearchContext.Provider>
  )
}
