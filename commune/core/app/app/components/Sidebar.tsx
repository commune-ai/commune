'use client'
import { useState, useEffect } from 'react'
import { Home, Package, Info, Mail, ChevronLeft, ChevronRight, Search, X } from 'lucide-react'

interface SidebarProps {
  isExpanded: boolean
  onToggleExpand: () => void
}

export const Sidebar = ({ isExpanded, onToggleExpand }: SidebarProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearchFocused, setIsSearchFocused] = useState(false)

  useEffect(() => {
    setIsAnimating(true)
    const timer = setTimeout(() => setIsAnimating(false), 300)
    return () => clearTimeout(timer)
  }, [isExpanded])

  const navItems = [
    { icon: Home, label: 'HOME', href: '/search' },
    { icon: Package, label: 'EXPLORER', href: '/modules' },
    { icon: Info, label: 'ABOUT', href: '/about' },
  ]

  const filteredNavItems = navItems.filter(item => 
    item.label.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div 
      className={`fixed left-0 top-0 h-full bg-black border-r border-green-500 z-50 transition-all duration-300 ${
        isExpanded ? 'w-64' : 'w-16'
      }`}
    >
      {/* Toggle Button */}
      <button
        onClick={onToggleExpand}
        className="absolute -right-3 top-20 bg-black border border-green-500 rounded-full p-1 hover:bg-green-500 hover:text-black transition-colors z-10"
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

      {/* Search Bar */}
      {isExpanded && (
        <div className="p-4 border-b border-green-500">
          <div className={`relative flex items-center border ${
            isSearchFocused ? 'border-green-400' : 'border-green-500'
          } rounded-lg transition-colors`}>
            <Search size={16} className="absolute left-3 text-green-500" />
            <input
              type="text"
              placeholder="SEARCH..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => setIsSearchFocused(true)}
              onBlur={() => setIsSearchFocused(false)}
              className="w-full bg-transparent text-green-500 placeholder-green-500/50 pl-10 pr-8 py-2 focus:outline-none font-mono text-sm uppercase"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 text-green-500 hover:text-green-400"
              >
                <X size={16} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="p-4">
        <ul className="space-y-4">
          {filteredNavItems.map((item) => {
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
          {filteredNavItems.length === 0 && isExpanded && (
            <li className="text-green-500/50 font-mono text-sm uppercase text-center">
              NO RESULTS
            </li>
          )}
        </ul>
      </nav>

    </div>
  )
}

export default Sidebar