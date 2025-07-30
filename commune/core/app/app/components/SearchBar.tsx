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
      <div className={`flex items-center gap-4 px-8 py-5 bg-black/90 border-2 rounded-full transition-all duration-300 ${
        isFocused ? 'border-green-400 shadow-2xl shadow-green-500/40 scale-105' : 'border-green-500/50 hover:border-green-500/70'
      }`}>
        <Search size={24} className="text-green-500/90" />
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          className="flex-1 bg-transparent text-green-400 text-lg placeholder-green-600/40 focus:outline-none font-mono tracking-wider"
        />
        {query && (
          <button
            type="button"
            onClick={handleClear}
            className="text-green-400 hover:text-green-300 transition-all duration-200 p-2 rounded-full hover:bg-green-500/20"
            aria-label="Clear search"
          >
            <X size={20} />
          </button>
        )}
        <button
          type="submit"
          className="px-6 py-3 text-green-400 hover:text-black hover:bg-green-500 transition-all duration-300 rounded-full border border-green-500/60 hover:border-green-500 font-mono uppercase tracking-wide"
          aria-label="Search"
        >
          <Search size={22} />
        </button>
      </div>
    </form>
  )
}

export default SearchBar