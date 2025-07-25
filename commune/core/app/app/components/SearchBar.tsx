'use client'
import { useState } from 'react'
import { Search, X } from 'lucide-react'

interface SearchBarProps {
  onSearch: (query: string) => void
  placeholder?: string
  className?: string
}

export const SearchBar = ({ onSearch, placeholder = 'search modules...', className = '' }: SearchBarProps) => {
  const [query, setQuery] = useState('')
  const [isFocused, setIsFocused] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSearch(query)
  }

  const handleClear = () => {
    setQuery('')
    onSearch('')
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setQuery(value)
    // Real-time search as user types
    onSearch(value)
  }

  return (
    <form onSubmit={handleSubmit} className={`relative ${className}`}>
      <div className={`flex items-center gap-2 px-4 py-2 bg-black/60 border rounded-lg transition-all ${
        isFocused ? 'border-green-400 shadow-lg shadow-green-500/20' : 'border-green-500/30'
      }`}>
        <Search size={18} className="text-green-500/70" />
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          className="flex-1 bg-transparent text-green-400 text-sm placeholder-green-600/50 focus:outline-none"
        />
        {query && (
          <button
            type="button"
            onClick={handleClear}
            className="text-green-400 hover:text-green-300 transition-colors"
            aria-label="Clear search"
          >
            <X size={16} />
          </button>
        )}
        <button
          type="submit"
          className="px-3 py-1 text-green-400 hover:text-green-300 hover:bg-green-500/10 transition-colors border-l border-green-500/30 pl-3"
          aria-label="Search"
        >
          <Search size={18} />
        </button>
      </div>
    </form>
  )
}

export default SearchBar