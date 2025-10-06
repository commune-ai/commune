'use client'
import { useState, useEffect } from 'react'
import { Home, Package, Info, ChevronLeft, ChevronRight, Search, X } from 'lucide-react'

interface SidebarProps {
  isExpanded: boolean
  onToggleExpand: () => void
}

export const Sidebar = ({ isExpanded, onToggleExpand }: SidebarProps) => {
  const [searchQuery, setSearchQuery] = useState('')

  const navItems = [
    { icon: Home, label: 'HOME', href: '/' },
    { icon: Package, label: 'MODULES', href: '/modules' },
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
      <button
        onClick={onToggleExpand}
        className="absolute -right-3 top-20 bg-black border border-green-500 rounded-full p-1 hover:bg-green-500 hover:text-black transition-colors z-10"
      >
        {isExpanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>

      <div className="p-4 border-b border-green-500">
        <h2 className={`text-green-500 font-mono uppercase transition-all duration-300 ${
          isExpanded ? 'text-xl' : 'text-sm text-center'
        }`}>
          {isExpanded ? 'NAVIGATION' : 'NAV'}
        </h2>
      </div>

      {isExpanded && (
        <div className="p-4 border-b border-green-500">
          <div className="relative flex items-center border border-green-500 rounded-lg">
            <Search size={16} className="absolute left-3 text-green-500" />
            <input
              type="text"
              placeholder="SEARCH..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
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
                    <span className="transition-opacity duration-300">
                      {item.label}
                    </span>
                  )}
                </a>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

export default Sidebar