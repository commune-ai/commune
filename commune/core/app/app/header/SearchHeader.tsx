'use client'
import { useState, useCallback } from 'react'
import { Search } from 'lucide-react'
import { useRouter, usePathname } from 'next/navigation'
import { useSearchContext } from '@/app/context/SearchContext'
export const SearchHeader = () => {
  const [isSearchExpanded, setIsSearchExpanded] = useState(false)
  const [localSearchTerm, setLocalSearchTerm] = useState('')
  const { searchFilters, updateSearchFilters, isSearchActive } = useSearchContext()
  const router = useRouter()
  const pathname = usePathname()
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    updateSearchFilters({ searchTerm: localSearchTerm })
    if (pathname !== '/') {
      router.push(`/`)
    }
  }

  const handleSearchChange = (value: string) => {
    setLocalSearchTerm(value)
  }

  return (
    <div className="flex items-center gap-2">
      {/* Search Form */}
      <form onSubmit={handleSearchSubmit} className="flex items-center gap-2 flex-1 max-w-md">
        <div className="relative flex-1">
          <input
            type="text"
            placeholder="SEARCH..."
            value={localSearchTerm}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="h-12 w-full px-4 text-sm bg-black border border-green-500 text-green-500 placeholder-green-500/50 focus:outline-none focus:border-green-400 focus:shadow-[0_0_10px_rgba(34,197,94,0.3)] hover:bg-green-500/10 hover:border-green-400 uppercase transition-all font-mono rounded-lg"
            style={{
              fontFamily: 'monospace',
              letterSpacing: '0.05em'
            }}
          />
          <Search className="absolute right-3 top-1/2 -translate-y-1/2 text-green-500/50" size={16} />
          {localSearchTerm.startsWith('/') && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-green-400 font-bold text-xs uppercase">
              CMD
            </div>
          )}
        </div>
      </form>
    </div>
  )
}

export default SearchHeader