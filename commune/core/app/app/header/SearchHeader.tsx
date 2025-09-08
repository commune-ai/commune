'use client'

import { useState, useCallback } from 'react'
import { Search } from 'lucide-react'
import { useRouter, usePathname } from 'next/navigation'
import { useSearchContext } from '@/app/context/SearchContext'

export const SearchHeader = () => {
  const [localSearchTerm, setLocalSearchTerm] = useState('')
  const { updateSearchFilters } = useSearchContext()
  const router = useRouter()
  const pathname = usePathname()

  const handleSearchSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      updateSearchFilters({ searchTerm: localSearchTerm, page: 1 })
      if (pathname !== '/') router.push(`/`)
    },
    [localSearchTerm, pathname, router, updateSearchFilters]
  )

  return (
    <div className="flex items-center gap-3">
      <form onSubmit={handleSearchSubmit} className="flex-1 max-w-md">
        <div className="relative group">
          <input
            type="text"
            inputMode="search"
            placeholder="TYPE QUERY OR /COMMAND"
            value={localSearchTerm}
            onChange={(e) => setLocalSearchTerm(e.target.value.trim().toLowerCase())}
            className="h-11 w-full rounded-lg bg-black font-mono text-sm uppercase tracking-wider
                       text-green-400 placeholder-green-700
                       border border-green-500 focus:outline-none
                       focus:border-green-400 focus:shadow-[0_0_8px_rgba(34,197,94,0.3)]
                       pl-10 pr-3 transition-all"
          />
          <Search
            className="absolute left-3 top-1/2 -translate-y-1/2 text-green-500/70 group-focus-within:text-green-400"
            size={16}
            aria-hidden
          />
        </div>
      </form>
    </div>
  )
}

export default SearchHeader
