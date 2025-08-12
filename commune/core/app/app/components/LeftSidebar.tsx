'use client'
import { useState, useEffect } from 'react'
import { Home, Package, Info, Mail, ChevronLeft, ChevronRight, Search, X } from 'lucide-react'

interface LeftSidebarProps {
  isExpanded: boolean
  onToggleExpand: () => void
}

export const LeftSidebar = ({ isExpanded, onToggleExpand }: LeftSidebarProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [advancedFilters, setAdvancedFilters] = useState({
    category: '',
    dateRange: '',
    sortBy: 'relevance'
  })

  useEffect(() => {
    setIsAnimating(true)
    const timer = setTimeout(() => setIsAnimating(false), 300)
    return () => clearTimeout(timer)
  }, [isExpanded])

  const navItems = [
    { icon: Home, label: 'HOME', href: '/' },
    { icon: Package, label: 'MODULES', href: '/modules' },
    { icon: Info, label: 'ABOUT', href: '/about' },
    { icon: Mail, label: 'CONTACT', href: '/contact' }
  ]

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    // Implement search logic here
    console.log('Searching for:', query, 'with filters:', advancedFilters)
  }

  return (
    <div 
      className={`fixed left-0 top-0 h-full bg-black border-r border-green-500 z-30 transition-all duration-300 ${
        isExpanded ? 'w-64' : 'w-16'
      }`}
    >
      {/* Toggle Button */}
      <button
        onClick={onToggleExpand}
        className="absolute -right-3 top-20 bg-black border border-green-500 rounded-full p-1 hover:bg-green-500 hover:text-black transition-colors"
      >
        {isExpanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>

      {/* Logo/Header */}
      <div className="p-4 border-b border-green-500">
        <h2 className={`text-green-500 font-mono uppercase transition-all duration-300 ${
          isExpanded ? 'text-xl' : 'text-sm text-center'
        }`}>
          {isExpanded ? 'NAVIGATION' : 'NAV'}
        </h2>
      </div>

      {/* Search Section */}
      {isExpanded && (
        <div className="p-4 border-b border-green-500">
          {/* Basic Search */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="SEARCH..."
              className="w-full px-3 py-2 bg-black border border-green-500 text-green-500 placeholder-green-500/50 font-mono text-sm focus:outline-none focus:border-green-400"
            />
            <Search className="absolute right-2 top-2.5 text-green-500/50" size={16} />
          </div>

          {/* Advanced Search Toggle */}
          <button
            onClick={() => setShowAdvancedSearch(!showAdvancedSearch)}
            className="mt-2 text-green-500 text-xs font-mono hover:text-green-400 transition-colors"
          >
            {showAdvancedSearch ? '- HIDE ADVANCED' : '+ ADVANCED SEARCH'}
          </button>

          {/* Advanced Search Options */}
          {showAdvancedSearch && (
            <div className="mt-3 space-y-2 p-3 border border-green-500/30 rounded">
              <div>
                <label className="text-green-500 text-xs font-mono block mb-1">CATEGORY</label>
                <select
                  value={advancedFilters.category}
                  onChange={(e) => setAdvancedFilters({...advancedFilters, category: e.target.value})}
                  className="w-full px-2 py-1 bg-black border border-green-500/50 text-green-500 text-xs font-mono focus:outline-none focus:border-green-400"
                >
                  <option value="">ALL</option>
                  <option value="modules">MODULES</option>
                  <option value="docs">DOCS</option>
                  <option value="api">API</option>
                </select>
              </div>

              <div>
                <label className="text-green-500 text-xs font-mono block mb-1">DATE RANGE</label>
                <select
                  value={advancedFilters.dateRange}
                  onChange={(e) => setAdvancedFilters({...advancedFilters, dateRange: e.target.value})}
                  className="w-full px-2 py-1 bg-black border border-green-500/50 text-green-500 text-xs font-mono focus:outline-none focus:border-green-400"
                >
                  <option value="">ANY TIME</option>
                  <option value="today">TODAY</option>
                  <option value="week">THIS WEEK</option>
                  <option value="month">THIS MONTH</option>
                  <option value="year">THIS YEAR</option>
                </select>
              </div>

              <div>
                <label className="text-green-500 text-xs font-mono block mb-1">SORT BY</label>
                <select
                  value={advancedFilters.sortBy}
                  onChange={(e) => setAdvancedFilters({...advancedFilters, sortBy: e.target.value})}
                  className="w-full px-2 py-1 bg-black border border-green-500/50 text-green-500 text-xs font-mono focus:outline-none focus:border-green-400"
                >
                  <option value="relevance">RELEVANCE</option>
                  <option value="date">DATE</option>
                  <option value="name">NAME</option>
                  <option value="popularity">POPULARITY</option>
                </select>
              </div>

              <button
                onClick={() => handleSearch(searchQuery)}
                className="w-full mt-2 px-2 py-1 bg-green-500/10 border border-green-500 text-green-500 text-xs font-mono hover:bg-green-500 hover:text-black transition-colors"
              >
                APPLY FILTERS
              </button>
            </div>
          )}
        </div>
      )}

      {/* Navigation */}
      <nav className="p-4">
        <ul className="space-y-4">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <li key={item.href}>
                <a 
                  href={item.href} 
                  className="flex items-center gap-3 text-green-500 hover:text-green-400 font-mono uppercase transition-colors"
                >
                  <Icon size={20} className="flex-shrink-0" />
                  {isExpanded && (
                    <span className={`transition-opacity duration-300 ${
                      isAnimating ? 'opacity-0' : 'opacity-100'
                    }`}>
                      {item.label}
                    </span>
                  )}
                </a>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Footer Info */}
      {isExpanded && (
        <div className="absolute bottom-4 left-4 right-4 border-t border-green-500 pt-4">
          <p className="text-green-500 text-xs font-mono opacity-50">
            COMMUNE AI
          </p>
        </div>
      )}
    </div>
  )
}

export default LeftSidebar