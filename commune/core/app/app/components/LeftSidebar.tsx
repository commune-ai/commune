'use client'
import { useState, useEffect } from 'react'
import { Home, Package, Info, Mail, ChevronLeft, ChevronRight } from 'lucide-react'

interface LeftSidebarProps {
  isExpanded: boolean
  onToggleExpand: () => void
}

export const LeftSidebar = ({ isExpanded, onToggleExpand }: LeftSidebarProps) => {
  const [isAnimating, setIsAnimating] = useState(false)

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